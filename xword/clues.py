import abc
import asyncio
from collections import Counter, defaultdict
import enum
import functools
import html
import io
import itertools
import json
import logging
import math
import random
import re
import os
import string
import sys
from typing import cast, Any, Awaitable, Dict, Iterable, List, Optional, Union
import traceback
import urllib.parse

import aiohttp
import bs4
import tqdm

from .utils import answerize, normalize_unicode, uncancel, GroupException, WasCancelledError

'''
Scoring ported and extended from crows project at https://github.com/kcaze/crows.
'''

ROOT = os.path.dirname(os.path.dirname(__file__))
CACHE_DIR = os.path.join(ROOT, 'cache')
TIMEOUT_SECONDS = 10
PROXY_TIMEOUT_SECONDS = 20

class Proxy(object):
	def __init__(self, raise_on_error=False):
		self.proxies = None # type: Optional[List[str]]
		self.lock = asyncio.Semaphore()
		self.raise_on_error = raise_on_error
		self.warned = False

	async def get_proxy(self, session : aiohttp.ClientSession) -> Optional[str]:
		if self.proxies is None:
			async with self.lock:
				if self.proxies is None:
					url = 'https://api.proxyscrape.com/?request=getproxies&proxytype=http&timeout=5000&country=all&ssl=yes&anonymity=elite'
					task = session.get(url, timeout=TIMEOUT_SECONDS)
					try:
						async with task as response:
							doc = await response.text()
					except Exception:
						traceback.print_exc()
						raise
					self.proxies = doc.split()
					logging.info('Got {} proxies.'.format(len(self.proxies)))
		proxy = None
		if self.proxies:
			idx = random.randrange(len(self.proxies))
			proxy = 'http://' + self.proxies[idx % len(self.proxies)]
		else:
			if self.raise_on_error:
				raise RuntimeError('Could not find proxies')
			else:
				if not self.warned:
					async with self.lock:
						if not self.warned:
							self.warned = True
							logging.warning('Could not get a proxies list. Fetching resources directly.')
		return proxy

async def gather_resolve_dict(d : Dict[Any, Awaitable[Any]], excs : Optional[GroupException] = None) -> Dict[Any, Any]:
	'''
	Resolves futures in dict values. Mutates the dict.

	Aggregates exceptions in excs.
	'''
	d_future = {}
	for key, value in d.items():
		d_future[key] = asyncio.ensure_future(value)
	await asyncio.wait(d_future.values())
	d_result = {}
	for key, value in d_future.items():
		try:
			d_result[key] = value.result()
		except Exception as e:
			d_result[key] = None
			if excs is not None:
				excs.exceptions.append(e)
	return d_result

class AsyncNoop(object):
	def __aenter__(self):
		return self
	def __await__(self):
		return iter(())
	def __aexit__(self, exc_type, exc_value, traceback):
		return self

with open(os.path.join(ROOT, 'data/google_ignore.txt')) as f:
	stopwords = set(line.strip() for line in f if line.strip() and not line.strip().startswith('#'))
nonalpha_regex = re.compile(r'\W+')
def get_text_ngrams(lines : List[str], values : Optional[Iterable[int]] = None) -> Dict[str, int]:
	'''Count the number of unigrams and bigrams in a series of results.'''
	counter = Counter() # type: Dict[str, int]
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
	num_trials = 3
	proxy_num_trials = 2
	min_valid_html_length = 1000 # for asserting we aren't getting error responses, valid responses seem to be at least 8k
	semaphore = AsyncNoop() # type: Union[AsyncNoop, asyncio.Semaphore]
	use_proxy = True
	parse_json = False
	site_gave_answers = False # fetch will set to True on returned results
	proxy_num_tasks = 5 # number of proxies to try for each clue
	redundant_fetch = False # set to True if redundant for parent class
	# subclasses should override
	expected_answers = True
	should_run = True # set to False if the query should not be run (eg, filters don't apply)
	fetch_fail = 0 # incremented when a fetch fails
	fetch_success = 0 # incremented when a fetch succeeds

	def __init__(self, clue : str, session : aiohttp.ClientSession, proxy : Proxy, length_guess : int, async_tqdm : Optional[tqdm.tqdm] = None):
		self.clue = clue
		self.session = session
		self.proxy = proxy
		# some sites require a length
		self.length_guess = length_guess
		self.async_tqdm = async_tqdm
		self.trial = 0

	@abc.abstractmethod
	def url(self) -> str:
		raise NotImplementedError()

	def formdata(self) -> aiohttp.FormData:
		raise NotImplementedError()

	def url_clue(self, dash = False, safe : str = '') -> str:
		text = self.clue
		if dash:
			text = re.sub(r'\s', '-', text)
		return urllib.parse.quote_plus(text, safe=safe)

	@staticmethod
	def slugify(s):
		s = normalize_unicode(s).lower()
		s = re.sub(r'\W+', '-', s)
		return s

	def is_valid(self, doc : str) -> bool:
		if len(doc) < self.min_valid_html_length:
			return False
		return True

	async def get_scores(self) -> Dict[str, float]:
		doc = await self.fetch()
		answer_scores = self._get_scores(doc)
		if answer_scores:
			self.__class__.site_gave_answers = True
		return answer_scores

	@abc.abstractmethod
	def _get_scores(self, doc : str) -> Dict[str, float]:
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
			trial = 0
			while trial < self.num_trials:
				if self.use_proxy and trial >= self.proxy_num_trials:
					break
				trial += 1
				self.trial = trial
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
					pending = set()
					if self.use_proxy:
						proxy = await self.proxy.get_proxy(self.session)
						if proxy is None:
							self.use_proxy = False
					num_tasks = self.proxy_num_tasks if self.use_proxy else 1
					for i in range(num_tasks):
						if self.use_proxy:
							proxy = await self.proxy.get_proxy(self.session)
							timeout = PROXY_TIMEOUT_SECONDS
						else:
							proxy = None
							timeout = TIMEOUT_SECONDS
						fetch_kwargs = {
							'timeout': timeout,
							'proxy': proxy,
						}
						coroutine = uncancel(task_maker(**fetch_kwargs))
						pending.add(asyncio.ensure_future(coroutine))
					semaphore = AsyncNoop() if self.use_proxy else self.semaphore
					async with semaphore:
						while doc is None and pending:
							try:
								done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
							except asyncio.CancelledError as e:
								for task in done | pending:
									task.cancel()
									try:
										await task
									except Exception:
										pass
								raise e
							for task in done:
								try:
									response = task.result()
									_bytes = await response.read()
									_doc = _bytes.decode('utf-8', 'ignore')
									if not self.is_valid(_doc):
										raise aiohttp.ClientError('Invalid response')
									if self.parse_json:
										# check for valid json
										json.loads(_doc)
									doc = _doc
								except aiohttp.ClientError:
									pass
								except json.decoder.JSONDecodeError:
									pass
								except asyncio.TimeoutError:
									pass
								except Exception:
									# show other errors but continue
									traceback.print_exc()
						for task in pending:
							task.cancel()
							try:
								await task
							except Exception:
								# we don't care about errors in cancelled tasks
								pass
					if doc is None:
						if not self.use_proxy:
							# wait and try others before trying again
							await asyncio.sleep(1)
					else:
						self.__class__.fetch_success += 1
						os.makedirs(CACHE_DIR, exist_ok=True)
						with open(cache_file, 'wb') as f:
							f.write(doc.encode('utf-8'))
				if doc is not None:
					return doc
			# could not get resource
			if not self.__class__.fetch_fail:
				logging.warning('Failed connection to {} after {} tries... skipping'.format(self.__class__.__name__, trial))
			self.__class__.fetch_fail += 1
			return ''
		finally:
			if self.async_tqdm is not None:
				self.async_tqdm.update()


class Tracker(enum.Enum):
	@classmethod
	async def get_scores_by_tracker(
		cls,
		clue : str,
		session : aiohttp.ClientSession,
		proxy : Proxy, length_guess : int,
		trackers : Optional[Iterable['Tracker']] = None,
		async_tqdm : Optional[tqdm.tqdm] = None,
		excs : Optional[GroupException] = None,
	) -> Dict[Any, Dict[str, float]]:
		scores = {
			tracker.value: tracker.value(clue, session, proxy, length_guess, async_tqdm=async_tqdm).get_scores()
			for tracker in (cls if trackers is None else trackers) # type: ignore # mypy does not recognize enums
		}
		tracker_scores = await gather_resolve_dict(scores, excs=excs)
		return tracker_scores

	@classmethod
	async def aggregate_scores(
		cls,
		clue : str,
		session : aiohttp.ClientSession,
		proxy : Proxy, length_guess : int,
		async_tqdm : Optional[tqdm.tqdm] = None,
		excs : Optional[GroupException] = None,
	) -> Dict[str, float]:
		'''
		Aggregate trackers and reduce by sum.

		Aggregates exceptions in excs
		'''
		tracker_scores = await cls.get_scores_by_tracker(clue, session, proxy, length_guess, async_tqdm=async_tqdm, excs=excs)
		scores = defaultdict(float) # type: Dict[str, float]
		for tracker, _scores in tracker_scores.items():
			# skip redundant trackers whose parent have results
			if tracker.redundant_fetch and tracker_scores[tracker.__bases__[0]]:
				continue
			if _scores is not None:
				for key, value in _scores.items():
					scores[key] += value
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
		def _get_scores(self, doc : str) -> Dict[str, float]:
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

		NB: The only difference in results is ACROSSNDOWN removes spaces.
		'''
		method = 'get'
		min_valid_html_length = 20000 # server outage is about 15k, valid response is about 150k
		semaphore = asyncio.Semaphore(6)
		star_tag = r'<div></div>'
		regex = re.compile(r'<tr[^>]*>\s*<td><div class=stars>((?:{})*)</div><div class=clear></div></td><td><a[^>]*>([\w\s]+)</a>'.format(star_tag))
		def url(self) -> str:
			return 'https://www.wordplays.com/crossword-solver/{}'.format(self.url_clue(dash=True))
		def is_valid(self, doc : str) -> bool:
			if not super().is_valid(doc):
				return False
			if type(self) == Tracker.WORDPLAYS.value: # type: ignore # mypy does not recognize enums
				if 'https://www.google.com/recaptcha/api.js' in doc:
					return False
			return True
		def _get_scores(self, doc : str) -> Dict[str, float]:
			answer_scores = {} # type: Dict[str, float]
			matches = self.regex.findall(doc)
			for match in matches:
				star_tags, answer = match
				stars = round(len(star_tags) / len(self.star_tag))
				value = stars
				answer_scores[answer] = value
			return answer_scores

	class ACROSSNDOWN(WORDPLAYS):
		semaphore = asyncio.Semaphore(6)
		# class counts
		fetch_fail = 0 # needs to have separate count from WORDPLAYS
		fetch_success = 0 # needs to have separate count from WORDPLAYS
		@property
		def redundant_fetch(self): # property because of class definition ordering
			return WORDPLAYS
		def url(self) -> str:
			return 'http://www.acrossndown.com/crosswords/clues/{}'.format(self.url_clue(dash=True, safe=string.punctuation))

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
		def _get_scores(self, doc : str) -> Dict[str, float]:
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
		def _get_scores(self, doc : str) -> Dict[str, float]:
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
		def _get_scores(self, doc : str) -> Dict[str, float]:
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
		def _get_scores(self, doc : str) -> Dict[str, float]:
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
		def _get_scores(self, doc : str) -> Dict[str, float]:
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
		def _get_scores(self, doc : str) -> Dict[str, float]:
			answer_scores = {} # Dict[str, float]
			completions = []
			try:
				query, results, _, _, extra = json.loads(doc) # type: ignore # mypy does not handle underscore # str, List[str], Any, Any, Dict[str, Any]
			except Exception:
				pass
			else:
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
		# class counts
		fetch_fail = 0 # needs to have separate count from GOOGLE
		fetch_success = 0 # needs to have separate count from GOOGLE
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
	proxy = Proxy(raise_on_error=True)
	clue = 'batman sidekick'
	clue = "The Metamorphosis author's first name... / ...and his last name"
	clue = 'Intro to physics? / "Life In Hell" character'
	# clue = '😠'
	q = []
	trackers = [Tracker.WORDPLAYS] # type: Optional[List[Any]] # mypy does not recognize enums
	trackers = None
	for i in range(1):
		q.append(Tracker.get_scores_by_tracker(clue, session, proxy, 5, trackers=trackers))
	all_tasks = asyncio.gather(*q)
	loop = asyncio.get_event_loop()
	answer_scores = loop.run_until_complete(all_tasks)
	loop.run_until_complete(session.close())
	loop.close()
	result = answer_scores[0]
	for tracker, values in result.items():
		print(tracker.__name__, values)
