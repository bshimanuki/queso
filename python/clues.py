import abc
import asyncio
from collections import Counter, defaultdict
import enum
import functools
import html
import io
import itertools
import random
import re
import os
from typing import cast, Any, Awaitable, Dict, Iterable, Optional, Union
import urllib.parse
import warnings

import aiohttp
import bs4

from utils import answerize, normalize_unicode

'''
Scoring ported and extended from crows project at https://github.com/kcaze/crows.
'''

ROOT = os.path.dirname(os.path.dirname(__file__))
CACHE_DIR = os.path.join(ROOT, 'cache')


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


class TrackerBase(abc.ABC):
	method = None # type: Optional[str]
	num_trials = 3
	min_valid_html_length = 1000 # for asserting we aren't getting error responses, valid responses seem to be at least 8k
	semaphore = AsyncNoop() # type: Union[AsyncNoop, asyncio.Semaphore]

	def __init__(self, clue : str, session : aiohttp.ClientSession, length_guess : int):
		self.clue = clue
		self.session = session
		# some sites require a length
		self.length_guess = length_guess
		self.trial = 0

	@abc.abstractmethod
	def url(self) -> str:
		raise NotImplementedError()

	def formdata(self) -> aiohttp.FormData:
		raise NotImplementedError()

	@abc.abstractmethod
	async def get_scores(self) -> Dict[str, float]:
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

	async def fetch(self) -> str:
		'''
		Return the html of the requested resource. Caches the result.
		'''
		assert self.method in ('get', 'post')
		url = self.url()
		for _trial in range(self.num_trials):
			self.trial = _trial
			if self.method == 'get':
				task_maker = lambda: self.session.get(url)
				cache_key = self.slugify('-'.join((self.method, url)))
			else:
				assert self.method == 'post'
				formdata = self.formdata()
				task_maker = lambda: self.session.post(url, data=formdata)
				class AsyncStreamWriter():
					def __init__(self):
						self.bufs = []
					async def write(self, s : bytes) -> None:
						self.bufs.append(s)
					def dump(self):
						return b''.join(self.bufs)
				formdump = AsyncStreamWriter()
				await formdata().write(formdump)
				cache_key = self.slugify('-'.join((self.method, url, formdump.dump().decode('utf-8'))))
			cache_file = os.path.join(CACHE_DIR, cache_key)
			if os.path.isfile(cache_file):
				with open(cache_file, 'rb') as f:
					doc = f.read().decode('utf-8')
			else:
				task = task_maker()
				async with self.semaphore:
					try:
						async with task as response:
							doc = await response.text()
						assert(len(doc) >= self.min_valid_html_length)
					except Exception as e:
						continue
					os.makedirs(CACHE_DIR, exist_ok=True)
					with open(cache_file, 'wb') as f:
						f.write(doc.encode('utf-8'))
			return doc
		warnings.warn('Failed connection to {} after {} tries... skipping'.format(self.__class__.__name__, self.num_trials))
		return ''


class Tracker(enum.Enum):
	@classmethod
	async def get_scores_by_tracker(cls, clue : str, session : aiohttp.ClientSession, length_guess : int, trackers : Optional[Iterable['Tracker']] = None) -> Dict[str, Dict[str, float]]:
		scores = {
			tracker.name: tracker.value(clue, session, length_guess).get_scores()
			for tracker in (cls if trackers is None else trackers)
		}
		tracker_scores = await gather_resolve_dict(scores)
		return tracker_scores

	@classmethod
	async def aggregate_scores(cls, clue : str, session : aiohttp.ClientSession, length_guess : int) -> Dict[str, float]:
		'''Aggregate trackers and reduce by max.'''
		tracker_scores = await cls.get_scores_by_tracker(clue, session, length_guess)
		scores = defaultdict(float) # type: Dict[str, float]
		for _scores in tracker_scores.values():
			for key, value in _scores.items():
				scores[key] = max(scores[key], value)
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
		async def get_scores(self) -> Dict[str, float]:
			doc = await self.fetch()
			answer_scores = {} # type: Dict[str, float]
			matches = self.regex.findall(doc)
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
		num_trials = 4 # try each site twice
		sites = [
			('https://www.wordplays.com/crossword-solver/{}', asyncio.Semaphore(12)),
			('http://www.acrossndown.com/crosswords/clues/{}', asyncio.Semaphore(12)),
		]
		star_tag = r'<div></div>'
		regex = re.compile(r'<tr[^>]*>\s*<td><div class=stars>((?:{})*)</div><div class=clear></div></td><td><a[^>]*>([\w\s]+)</a>'.format(star_tag))
		@property
		def semaphore(self):
			return self.sites[self.index][1]
		@property
		def index(self):
			return (self.trial + self.trial_offset) % len(self.sites)
		def __init__(self, clue : str, session : aiohttp.ClientSession, length_guess : int):
			super().__init__(clue, session, length_guess)
			self.trial_offset = random.randrange(len(self.sites))
		def url(self) -> str:
			site = self.sites[self.index][0]
			return site.format(self.url_clue(dash=True))
		async def get_scores(self) -> Dict[str, float]:
			doc = await self.fetch()
			answer_scores = {} # type: Dict[str, float]
			matches = self.regex.findall(doc)
			for match in matches:
				star_tags, answer = match
				stars = len(star_tags) // len(self.star_tag)
				value = 1 if stars >= 4 else 0
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
		async def get_scores(self) -> Dict[str, float]:
			doc = await self.fetch()
			answer_scores = {} # type: Dict[str, float]
			matches = self.regex.findall(doc)
			for match in matches:
				star_tags, star_tag, answer = match
				stars = round(len(star_tags) / len(star_tag))
				value = 1 if stars >= 4 else 0
				answer_scores[answer] = value
			return answer_scores

	class CROSSWORDNEXUS(TrackerBase):
		method = 'get'
		star_tag = r'<img src="/images/star.png" />'
		regex = re.compile(r'<tr[^>]*>\s*<td[^>]*>\s*((?:{})*)\s*</td>\s*<td[^>]*>\s*<big>\s*<a[^>]*>([\w\s]+)</a>'.format(star_tag))
		def url(self) -> str:
			return 'https://crosswordnexus.com/finder.php?clue={}'.format(self.url_clue())
		async def get_scores(self) -> Dict[str, float]:
			doc = await self.fetch()
			answer_scores = {} # type: Dict[str, float]
			matches = self.regex.findall(doc)
			for match in matches:
				star_tags, answer = match
				stars = len(star_tags) // len(self.star_tag)
				value = 1 if stars >= 3 else 0
				answer_scores[answer] = value
			return answer_scores

	class CROSSWORDTRACKER(TrackerBase):
		method = 'get'
		regex = re.compile(r'<li class="answer" data-count="(\d+)" data-text="([\w\s]+)">')
		def url(self) -> str:
			return 'http://crosswordtracker.com/search?clue={}'.format(self.url_clue(dash=True))
		async def get_scores(self) -> Dict[str, float]:
			doc = await self.fetch()
			answer_scores = {} # type: Dict[str, float]
			matches = self.regex.findall(doc)
			for match in matches:
				count, answer = match
				answer_scores[answer] = 1
			return answer_scores

	class ONEACROSS(TrackerBase):
		# TODO: determine if needs to be rate limited
		method = 'get'
		dot = '<img[^>]* src=[^>]*dot[^>]*>'
		star = '<img[^>]* src=[^>]*star[^>]*>'
		regex = re.compile('<tr>\s*<td[^>]*>\s*(?:{}\s*)*(({}\s*)*)</td>\s*<td[^>]*>\s*<tt>\s*<a[^>]*>([^<]*)</a>'.format(dot, star))
		def url(self) -> str:
			return 'http://www.oneacross.com/cgi-bin/search_banner.cgi?c0={}&p0={}'.format(self.url_clue(), self.length_guess)
		async def get_scores(self) -> Dict[str, float]:
			doc = await self.fetch()
			answer_scores = {} # type: Dict[str, float]
			matches = self.regex.findall(doc)
			for match in matches:
				star_tags, star_tag, answer = match
				stars = round(len(star_tags) / len(star_tag))
				value = 1 if stars >= 4 else 0
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
		nonalpha_regex = re.compile(r'\W+')
		with open(os.path.join(ROOT, 'google_ignore.txt')) as f:
			stopwords = set(line.strip() for line in f if line.strip() and not line.strip().startswith('#'))
		def url(self) -> str:
			return 'https://www.google.com/search?q={}'.format(self.url_clue())
		async def get_scores(self) -> Dict[str, float]:
			doc = await self.fetch()
			# remove bold tags
			doc = doc.replace('<em>', '').replace('</em>', '')
			doc = doc.replace('<b>', '').replace('</b>', '')

			wiki_scores = {} # type: Dict[str, float]
			wiki_matches = self.wikipeidia_regex.findall(doc)
			for wiki_match in wiki_matches:
				answer = answerize(wiki_match)
				wiki_scores[answer] = 1

			def get_text_ngrams(lines):
				counter = Counter()
				for line in lines:
					line = html.unescape(line)
					clusters = line.split()
					word_unigrams = set()
					word_bigrams = set()
					for i, cluster in enumerate(clusters):
						if cluster.endswith('\'s') or cluster.endswith('\'S'):
							# do once with the s appended and once without
							groups = (cluster[:-2], cluster[:-2] + 's')
						else:
							groups = (cluster,)
						for group in groups:
							words = self.nonalpha_regex.split(group)
							for word in words:
								word = answerize(word)
								if word and word not in self.stopwords:
									word_unigrams.add(word)
						if i + 1 < len(clusters):
							next_cluster = clusters[i + 1]
							pre = answerize(cluster)
							post = answerize(next_cluster)
							# allow stopwords to be in phrase but not at the end
							if pre and post and post not in self.stopwords:
								word_bigrams.add(pre + post)
					counter.update(word_unigrams)
					counter.update(word_bigrams)
				return counter

			header_matches = self.header_regex.findall(doc)
			snippet_matches = self.snippet_regex.findall(doc)
			results_counter = get_text_ngrams(header_matches + snippet_matches)
			results_scores = {} # type: Dict[str, float]
			for answer, count in results_counter.items():
				if count >= 4:
					value = 1
				else:
					value = 0
				results_scores[answer] = value

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
				soup_scores[answer] = 0

			answer_scores = {} # type: Dict[str, float]
			for answer in set(itertools.chain(wiki_scores.keys(), results_scores.keys(), soup_scores.keys())):
				answer_scores[answer] = 0
				if answer in wiki_scores and answer_scores[answer] < wiki_scores[answer]:
					answer_scores[answer] = wiki_scores[answer]
				if answer in results_scores and answer_scores[answer] < results_scores[answer]:
					answer_scores[answer] = results_scores[answer]
				if answer in soup_scores and answer_scores[answer] < soup_scores[answer]:
					answer_scores[answer] = soup_scores[answer]
			return answer_scores

		# TODO: class GOOGLE_CROSSWORD_CLUE(TrackerBase)

		# TODO: class GOOGLE_COMPLETION(TrackerBase)

		# TODO: class GOOGLE_FILL_IN_THE_BLANK(TrackerBase)


if __name__ == '__main__':
	headers = {
		'User-Agent': 'queso AppleWebKit/9000 Chrome/9000',
	}
	session = aiohttp.ClientSession(headers=headers)
	clue = 'batman sidekick'
	q = []
	trackers = [Tracker.WORDPLAYS]
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
