import abc
import asyncio
from collections import Counter, defaultdict
import enum
import functools
import html
import io
import itertools
import json
import math
import random
import re
import os
import sys
from typing import cast, Any, Awaitable, Counter as _Counter, Dict, Iterable, List, Optional, Union
import urllib.parse
import warnings

import aiohttp
import bs4
import tqdm

from utils import answerize, normalize_unicode

'''
Scoring ported and extended from crows project at https://github.com/kcaze/crows.
'''

ROOT = os.path.dirname(os.path.dirname(__file__))
CACHE_DIR = os.path.join(ROOT, 'cache')
TIMEOUT_SECONDS = 10
PROXY_TIMEOUT_SECONDS = 20

async def get_proxy(session : aiohttp.ClientSession, state={}) -> Optional[str]:
	if not state:
		state['proxies'] = None
		state['lock'] = asyncio.Semaphore()
	if state['proxies'] is None:
		async with state['lock']:
			if state['proxies'] is None:
				url = 'https://api.proxyscrape.com/?request=getproxies&proxytype=http&timeout=1000&country=all&ssl=yes&anonymity=elite'
				task = session.get(url, timeout=aiohttp.ClientTimeout(total=TIMEOUT_SECONDS))
				async with task as response:
					try:
						doc = await response.text()
					except:
						doc = ''
				state['proxies'] = doc.split()
	proxy = None
	if state['proxies']:
		idx = random.randrange(len(state['proxies']))
		proxy = 'http://' + state['proxies'][idx % len(state['proxies'])]
	else:
		warnings.warn('Could not get a proxies list.')
	return proxy

async def gather_resolve_dict(d : Dict[Any, Awaitable[Any]]) -> Dict[Any, Any]:
	'''Resolves futures in dict values. Mutates the dict.'''
	for key, value in d.items():
		d[key] = asyncio.ensure_future(value)
	d_future = cast(Dict[Any, 'asyncio.Future[Any]'], d)
	await asyncio.wait(d_future.values())
	d_result = cast(Dict[Any, Any], d)
	for key, value in d_future.items():
		d_result[key] = value.result()
	return d_result

class AsyncNoop(object):
	def __aenter__(self):
		return self
	def __await__(self):
		return iter(())
	def __aexit__(self, exc_type, exc_value, traceback):
		return self

with open(os.path.join(ROOT, 'google_ignore.txt')) as f:
	stopwords = set(line.strip() for line in f if line.strip() and not line.strip().startswith('#'))
nonalpha_regex = re.compile(r'\W+')
def get_text_ngrams(lines : List[str], values : Optional[Iterable[int]] = None) -> Dict[str, int]:
	'''Count the number of unigrams and bigrams in a series of results.'''
	counter = Counter() # type: _Counter[str]
	if values is None:
		values = (1 for line in lines)
	for line, value in zip(lines, values):
		line = html.unescape(line)
		clusters = line.split()
		word_unigrams = {}
		word_bigrams = {}
		for i, cluster in enumerate(clusters):
			if cluster.endswith('\'s') or cluster.endswith('\'S'):
				# do once with the s appended and once without
				groups = (cluster[:-2], cluster[:-2] + 's') # type: Iterable[str]
			else:
				groups = (cluster,)
			for group in groups:
				words = nonalpha_regex.split(group)
				for word in words:
					word = answerize(word)
					if word and word not in stopwords:
						word_unigrams[word] = value
			if i + 1 < len(clusters):
				next_cluster = clusters[i + 1]
				pre = answerize(cluster)
				post = answerize(next_cluster)
				# allow stopwords to be in phrase but not at the end
				if pre and post and post not in stopwords:
					word_bigrams[pre + post] = value
		counter.update(word_unigrams)
		counter.update(word_bigrams)
	return counter


class TrackerBase(abc.ABC):
	method = None # type: Optional[str]
	# num_trials = 3
	num_trials = 1
	min_valid_html_length = 1000 # for asserting we aren't getting error responses, valid responses seem to be at least 8k
	semaphore = AsyncNoop() # type: Union[AsyncNoop, asyncio.Semaphore]
	# use_proxy = False
	use_proxy = True
	parse_json = False
	# fetch will set to True on fetched resource
	site_gave_answers = False
	# subclasses should override
	expected_answers = True
	should_run = True
	could_not_fetch = 0

	def __init__(self, clue : str, session : aiohttp.ClientSession, length_guess : int, async_tqdm : Optional[tqdm.tqdm] = None):
		self.clue = clue
		self.session = session
		# some sites require a length
		self.length_guess = length_guess
		self.async_tqdm = async_tqdm
		self.trial = 0

	@abc.abstractmethod
	def url(self) -> str:
		raise NotImplementedError()

	def formdata(self) -> aiohttp.FormData:
		raise NotImplementedError()

	def url_clue(self, dash=False) -> str:
		text = self.clue
		if dash:
			text = re.sub(r'\s', '-', text)
		return urllib.parse.quote_plus(text)

	@staticmethod
	def slugify(s):
		s = normalize_unicode(s).lower()
		s = re.sub(r'\W+', '-', s)
		return s

	async def get_scores(self) -> Dict[str, float]:
		doc = await self.fetch()
		answer_scores = await self._get_scores(doc)
		if answer_scores:
			self.__class__.site_gave_answers = True
		return answer_scores

	@abc.abstractmethod
	async def _get_scores(self, doc : str) -> Dict[str, float]:
		raise NotImplementedError()

	async def fetch(self) -> str:
		'''
		Return the html of the requested resource. Caches the result.
		'''
		try:
			assert self.method in ('get', 'post')
			doc = None
			if not self.should_run:
				return ''
			for _trial in range(self.num_trials):
				self.trial = _trial
				url = self.url()
				if self.method == 'get':
					task_maker = lambda **fetch_kwargs: self.session.get(url, **fetch_kwargs)
					cache_key = self.slugify('-'.join((self.method, url)))
				else:
					assert self.method == 'post'
					formdata = self.formdata()
					task_maker = lambda **fetch_kwargs: self.session.post(url, data=formdata, **fetch_kwargs)
					class AsyncStreamWriter():
						def __init__(self):
							self.bufs = []
						async def write(self, s : bytes) -> None:
							self.bufs.append(s)
						def dump(self):
							return b''.join(self.bufs)
					formdump = AsyncStreamWriter()
					await formdata().write(formdump) # type: ignore # mock class
					cache_key = self.slugify('-'.join((self.method, url, formdump.dump().decode('utf-8'))))
				cache_file = os.path.join(CACHE_DIR, cache_key)
				if os.path.isfile(cache_file):
					with open(cache_file, 'rb') as f:
						doc = f.read().decode('utf-8')
				else:
					pending = []
					use_proxy = self.use_proxy
					if use_proxy:
						proxy = await get_proxy(self.session)
						if proxy is None:
							use_proxy = False
					num_tasks = 5 if use_proxy else 1
					for i in range(num_tasks):
						if use_proxy:
							proxy = await get_proxy(self.session)
							timeout = PROXY_TIMEOUT_SECONDS
						else:
							proxy = None
							timeout = TIMEOUT_SECONDS
						fetch_kwargs = {
							'timeout': aiohttp.ClientTimeout(total=timeout),
							'proxy': proxy,
						}
						task = task_maker(**fetch_kwargs)
						pending.append(asyncio.ensure_future(task))
					semaphore = AsyncNoop() if use_proxy else self.semaphore
					async with semaphore:
						while doc is None and pending:
							done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
							for task in done:
								try:
									response = await task
									_doc = await response.text()
									assert(len(_doc) >= self.min_valid_html_length)
									if self.parse_json:
										# check for valid json
										json.loads(_doc)
									doc = _doc
								except Exception:
									pass
						for task in pending:
							task.cancel()
							try:
								await task
							except:
								pass
					if doc is None:
						# wait and try others before trying again
						await asyncio.sleep(1)
					else:
						os.makedirs(CACHE_DIR, exist_ok=True)
						with open(cache_file, 'wb') as f:
							f.write(doc.encode('utf-8'))
				if doc is not None:
					return doc
			# could not get resource
			warnings.warn('Failed connection to {} after {} tries... skipping'.format(self.__class__.__name__, self.num_trials))
			self.__class__.could_not_fetch += 1
			return ''
		finally:
			if self.async_tqdm is not None:
				self.async_tqdm.update()


class Tracker(enum.Enum):
	@classmethod
	async def get_scores_by_tracker(cls, clue : str, session : aiohttp.ClientSession, length_guess : int, trackers : Optional[Iterable['Tracker']] = None, async_tqdm : Optional[tqdm.tqdm] = None) -> Dict[str, Dict[str, float]]:
		scores = {
			tracker.name: tracker.value(clue, session, length_guess, async_tqdm=async_tqdm).get_scores()
			for tracker in (cls if trackers is None else trackers) # type: ignore # mypy does not recognize enums
		}
		tracker_scores = await gather_resolve_dict(scores)
		return tracker_scores

	@classmethod
	async def aggregate_scores(cls, clue : str, session : aiohttp.ClientSession, length_guess : int, async_tqdm : Optional[tqdm.tqdm] = None) -> Dict[str, float]:
		'''Aggregate trackers and reduce by sum.'''
		tracker_scores = await cls.get_scores_by_tracker(clue, session, length_guess, async_tqdm=async_tqdm)
		scores = defaultdict(float) # type: Dict[str, float]
		for _scores in tracker_scores.values():
			for key, value in _scores.items():
				scores[key] += value
		# for key, value in scores.items():
			# scores[key] = math.sqrt(value)
		return scores


	class THECROSSWORDSOLVER(TrackerBase):
		method = 'post'
		regex = re.compile(r'<p class="searchresult"><a[^>]*>([\w\s]+)</a><span class="matchtypes">(?:<span[^>]*>[^<]*</span>\s*)*</span>')
		def url(self) -> str:
			return 'http://www.the-crossword-solver.com/search'
		def formdata(self) -> aiohttp.FormData:
			formdata = aiohttp.FormData()
			formdata.add_field('q', self.clue)
			return formdata
		async def _get_scores(self, doc : str) -> Dict[str, float]:
			answer_scores = {} # type: Dict[str, float]
			matches = self.regex.findall(doc)
			if any(c.isalpha() for c in self.clue):
				for match in matches:
					answer_scores[match] = 1
			return answer_scores

	class WORDPLAYS(TrackerBase):
		'''
		WORDPLAYS and ACROSSNDOWN use the same server. Both sites limit the number of actie connections.
		Requires user agent to be set to non-default.

		NB: This is nondeterministic because ACROSSNDOWN removes spaces.
		'''
		method = 'get'
		num_trials = 2 # try each site once (with multiple proxies each)
		min_valid_html_length = 20000 # server outage is about 15k, valid response is about 150k
		use_proxy = True
		sites = [
			('https://www.wordplays.com/crossword-solver/{}', asyncio.Semaphore(6)),
			('http://www.acrossndown.com/crosswords/clues/{}', asyncio.Semaphore(6)),
		]
		star_tag = r'<div></div>'
		regex = re.compile(r'<tr[^>]*>\s*<td><div class=stars>((?:{})*)</div><div class=clear></div></td><td><a[^>]*>([\w\s]+)</a>'.format(star_tag))
		@property
		def semaphore(self):
			return self.sites[self.index][1]
		@property
		def index(self):
			return (self.trial + self.trial_offset) % len(self.sites)
		def __init__(self, clue : str, session : aiohttp.ClientSession, length_guess : int, async_tqdm : Optional[tqdm.tqdm] = None):
			super().__init__(clue, session, length_guess, async_tqdm=async_tqdm)
			self.trial_offset = random.randrange(len(self.sites))
		def url(self) -> str:
			site = self.sites[self.index][0]
			return site.format(self.url_clue(dash=True))
		async def _get_scores(self, doc : str) -> Dict[str, float]:
			answer_scores = {} # type: Dict[str, float]
			matches = self.regex.findall(doc)
			for match in matches:
				star_tags, answer = match
				stars = round(len(star_tags) / len(self.star_tag))
				# value = 1 if stars >= 4 else 0
				value = stars
				answer_scores[answer] = value
			return answer_scores

	class CROSSWORDHEAVEN(TrackerBase):
		method = 'get'
		star_tag = r'^\s*<img[^>]*/>\s*$[\n\r]*'
		regex = re.compile(
			# r'<tr[^>]*>\s*<td[^>]*>\s*(({})*)\s*</td>\s*<td>\s*<a[^>]*>([\w\s]+)</a>'.format(star_tag), # desktop
			r'<tr[^>]*>\s*<td[^>]*>\s*(({})*)\s*</td>\s*<td>([\w\s]+)</td>'.format(star_tag), # mobile
			flags=re.MULTILINE,
		)
		def url(self) -> str:
			return 'https://m.crosswordheaven.com/search/result?clue={}'.format(self.url_clue())
		async def _get_scores(self, doc : str) -> Dict[str, float]:
			answer_scores = {} # type: Dict[str, float]
			matches = self.regex.findall(doc)
			for match in matches:
				star_tags, star_tag, answer = match
				stars = round(len(star_tags) / len(star_tag))
				# value = 1 if stars >= 4 else 0
				value = stars
				answer_scores[answer] = value
			return answer_scores

	class CROSSWORDNEXUS(TrackerBase):
		method = 'get'
		star_tag = r'<img src="/images/star.png" />'
		regex = re.compile(r'<tr[^>]*>\s*<td[^>]*>\s*((?:{})*)\s*</td>\s*<td[^>]*>\s*<big>\s*<a[^>]*>([\w\s]+)</a>'.format(star_tag))
		def url(self) -> str:
			return 'https://crosswordnexus.com/finder.php?clue={}'.format(self.url_clue())
		async def _get_scores(self, doc : str) -> Dict[str, float]:
			answer_scores = {} # type: Dict[str, float]
			matches = self.regex.findall(doc)
			for match in matches:
				star_tags, answer = match
				stars = len(star_tags) // len(self.star_tag)
				# value = 1 if stars >= 3 else 0
				value = stars / 3
				answer_scores[answer] = value
			return answer_scores

	class CROSSWORDTRACKER(TrackerBase):
		method = 'get'
		regex = re.compile(r'<li class="answer" data-count="(\d+)" data-text="([\w\s]+)">')
		def url(self) -> str:
			return 'http://crosswordtracker.com/search?clue={}'.format(self.url_clue(dash=True))
		async def _get_scores(self, doc : str) -> Dict[str, float]:
			answer_scores = {} # type: Dict[str, float]
			matches = self.regex.findall(doc)
			for match in matches:
				count, answer = match
				answer_scores[answer] = 1
			return answer_scores

	class ONEACROSS(TrackerBase):
		# TODO: determine if needs to be rate limited
		method = 'get'
		semaphore = asyncio.Semaphore(10)
		dot = '<img[^>]* src=[^>]*dot[^>]*>'
		star = '<img[^>]* src=[^>]*star[^>]*>'
		regex = re.compile('<tr>\s*<td[^>]*>\s*(?:{}\s*)*(({}\s*)*)</td>\s*<td[^>]*>\s*<tt>\s*<a[^>]*>([^<]*)</a>'.format(dot, star))
		def url(self) -> str:
			return 'http://www.oneacross.com/cgi-bin/search_banner.cgi?c0={}&p0={}'.format(self.url_clue(), self.length_guess)
		async def _get_scores(self, doc : str) -> Dict[str, float]:
			answer_scores = {} # type: Dict[str, float]
			matches = self.regex.findall(doc)
			for match in matches:
				star_tags, star_tag, answer = match
				stars = round(len(star_tags) / len(star_tag))
				# value = 1 if stars >= 4 else 0
				value = stars
				answer_scores[answer] = value
			return answer_scores

	class GOOGLE(TrackerBase):
		'''
		Requires User-Agent to have high enough versions of AppleWebKit and Chrome to get easier tags to parse.

		Google also locks out after too many requests in a short period.
		'''
		method = 'get'
		semaphore = asyncio.Semaphore(15)
		wikipeidia_regex = re.compile(r'>([^<]*) - Wikipedia<')
		header_regex = re.compile(r'<h3[^>]*>([^<]*)</h3>')
		snippet_regex = re.compile(r'<span [^>]*class="st"[^>]*>\s*(?:<span [^>]*class="f"[^>]*>[^<]*</span>)?\s*(.*?)</span>')
		def url(self) -> str:
			return 'https://www.google.com/search?q={}'.format(self.url_clue())
		async def _get_scores(self, doc : str) -> Dict[str, float]:
			# remove bold tags
			doc = doc.replace('<em>', '').replace('</em>', '')
			doc = doc.replace('<b>', '').replace('</b>', '')

			wiki_scores = {} # type: Dict[str, float]
			wiki_matches = self.wikipeidia_regex.findall(doc)
			wiki_counter = get_text_ngrams(wiki_matches)
			for answer, count in wiki_counter.items():
				wiki_scores[answer] = 3

			header_matches = self.header_regex.findall(doc)
			snippet_matches = self.snippet_regex.findall(doc)
			results_counter = get_text_ngrams(header_matches + snippet_matches)
			results_scores = {} # type: Dict[str, float]
			for answer, count in results_counter.items():
				results_scores[answer] = math.sqrt(count)

			soup = bs4.BeautifulSoup(doc, 'html.parser')
			for block in soup(['script', 'style']):
				block.decompose()
			for block in soup.find_all(class_='f'):
				block.decompose()
			# strings = list(soup.find(id='rcnt').stripped_strings)
			strings = list(soup.stripped_strings)
			soup_counter = get_text_ngrams(strings)
			soup_scores = {} # type: Dict[str, float]
			for answer, count in soup_counter.items():
				soup_scores[answer] = 1

			answer_scores = {} # type: Dict[str, float]
			for answer in set(itertools.chain(wiki_scores.keys(), results_scores.keys(), soup_scores.keys())):
				answer_scores[answer] = 1
				if answer in wiki_scores and answer_scores[answer] < wiki_scores[answer]:
					answer_scores[answer] = wiki_scores[answer]
				if answer in results_scores and answer_scores[answer] < results_scores[answer]:
					answer_scores[answer] = results_scores[answer]
				if answer in soup_scores and answer_scores[answer] < soup_scores[answer]:
					answer_scores[answer] = soup_scores[answer]
			return answer_scores

	# TODO: class GOOGLE_CROSSWORD_CLUE(TrackerBase)

	class GOOGLE_COMPLETION(TrackerBase):
		method = 'get'
		semaphore = asyncio.Semaphore(15)
		regex_word = re.compile(r'[A-Za-z]+')
		min_valid_html_length = 100 # json responses are short, especially when empty
		parse_json = True
		def url(self) -> str:
			return 'http://suggestqueries.google.com/complete/search?client=chrome&q={}'.format(self.url_clue())
		async def _get_scores(self, doc : str) -> Dict[str, float]:
			answer_scores = {} # Dict[str, float]
			completions = []
			query, results, _, _, extra = json.loads(doc) # type: ignore # mypy does not handle underscore # str, List[str], Any, Any, Dict[str, Any]
			results_relevance = extra.get('google:suggestrelevance', [500 for result in results]) # values in the hundreds and thousands
			query_relevance = extra.get('google:verbatimrelevance', 500) # values in the hundreds and thousands
			query = normalize_unicode(query).lower()
			query_words = self.regex_word.findall(query)
			for result in results:
				result = normalize_unicode(result).lower()
				for word  in query_words:
					result = result.replace(word, '', 1)
				completions.append(result)
			completions_counter = get_text_ngrams(completions, values=results_relevance)
			for answer, count in completions_counter.items():
				answer_scores[answer] = max(1, math.log2(count / 100))
			return answer_scores

	class GOOGLE_FILL_IN_THE_BLANK(GOOGLE):
		regex_blank = re.compile(r'_+')
		expected_answers = False
		@property
		def should_run(self):
			return '_' in self.clue
		def url(self) -> str:
			return self.regex_blank.sub('*', super().url())


if __name__ == '__main__':
	headers = {
		'User-Agent': 'queso AppleWebKit/9000 Chrome/9000',
	}
	session = aiohttp.ClientSession(headers=headers)
	clue = 'batman sidekick'
	# clue = '😠'
	q = []
	trackers = [Tracker.WORDPLAYS] # type: Optional[List[Any]] # mypy does not recognize enums
	trackers = None
	for i in range(1):
		q.append(Tracker.get_scores_by_tracker(clue, session, 5, trackers=trackers))
	all_tasks = asyncio.gather(*q)
	loop = asyncio.get_event_loop()
	answer_scores = loop.run_until_complete(all_tasks)
	loop.run_until_complete(session.close())
	loop.close()
	result = answer_scores[0]
	for tracker, values in result.items():
		print(tracker, values)
