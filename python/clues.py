import asyncio
from collections import defaultdict
import enum
import functools
import random
import re
import urllib.parse
import warnings

import aiohttp
from typing import cast, Any, Awaitable, Dict, Optional, Union

'''
Scoring ported from crows project at https://github.com/kcaze/crows.
'''


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


class TrackerBase(object):
	method = None # type: Optional[str]
	num_trials = 3
	semaphore = AsyncNoop() # type: Union[AsyncNoop, asyncio.Semaphore]
	def __init__(self, clue : str, session : aiohttp.ClientSession):
		self.clue = clue
		self.session = session
		self.trial = 0
	def url(self) -> str:
		raise NotImplementedError()
	def formdata(self) -> aiohttp.FormData:
		raise NotImplementedError()
	async def get_scores(self) -> Dict[str, float]:
		raise NotImplementedError()
	def url_clue(self, dash=False) -> str:
		text = self.clue
		if dash:
			text = re.sub('\s', '-', text)
		return urllib.parse.quote(text)
	async def fetch(self) -> str:
		assert self.method in ('get', 'post')
		url = self.url()
		for _trial in range(self.num_trials):
			self.trial = _trial
			if self.method == 'get':
				task = self.session.get(url)
			else:
				assert self.method == 'post'
				formdata = self.formdata()
				task = self.session.post(url, data=formdata)
			async with self.semaphore:
				try:
					async with task as response:
						html = await response.text()
					return html
				except:
					pass
		warnings.warn('Failed connection to {} after {} tries... skipping'.format(self.__name__, self.num_trials))
		return ''


class Tracker(enum.Enum):
	@classmethod
	async def get_scores_by_tracker(cls, clue : str, session : aiohttp.ClientSession) -> Dict[str, Dict[str, float]]:
		scores = {
			tracker.name: tracker.value(clue, session).get_scores()
			for tracker in cls
		}
		tracker_scores = await gather_resolve_dict(scores)
		return tracker_scores

	@classmethod
	async def aggregate_scores(cls, clue : str, session : aiohttp.ClientSession) -> Dict[str, float]:
		'''Aggregate trackers and reduce by max.'''
		tracker_scores = await cls.get_scores_by_tracker(clue, session)
		scores = defaultdict(float) # type: Dict[str, float]
		for _scores in tracker_scores.values():
			for key, value in _scores.items():
				scores[key] = max(scores[key], value)
		return scores


	class THECROSSWORDSOLVER(TrackerBase):
		method = 'post'
		def url(self) -> str:
			return 'http://www.the-crossword-solver.com/search'
		def formdata(self) -> aiohttp.FormData:
			formdata = aiohttp.FormData()
			formdata.add_field('q', self.clue)
			return formdata
		async def get_scores(self) -> Dict[str, float]:
			html = await self.fetch()
			answer_scores = {} # type: Dict[str, float]
			regex = r'<p class="searchresult"><a[^>]*>([\w\s]+)</a><span class="matchtypes">(?:<span[^>]*>[^<]*</span>\s*)*</span>'
			matches = re.findall(regex, html)
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
		@property
		def semaphore(self):
			return self.sites[self.index][1]
		@property
		def index(self):
			return (self.trial + self.trial_offset) % len(self.sites)
		def __init__(self, clue : str, session : aiohttp.ClientSession):
			super().__init__(clue, session)
			self.trial_offset = random.randrange(len(self.sites))
		def url(self) -> str:
			site = self.sites[self.index][0]
			return site.format(self.url_clue(dash=True))
		async def get_scores(self) -> Dict[str, float]:
			html = await self.fetch()
			answer_scores = {} # type: Dict[str, float]
			star_tag = r'<div></div>'
			regex = r'<tr[^>]*>\s*<td><div class=stars>((?:{})*)</div><div class=clear></div></td><td><a[^>]*>([\w\s]+)</a>'.format(star_tag)
			matches = re.findall(regex, html)
			for match in matches:
				star_tags, answer = match
				stars = len(star_tags) // len(star_tag)
				value = 1 if stars >= 4 else 0
				answer_scores[answer] = value
			return answer_scores

	class CROSSWORDNEXUS(TrackerBase):
		method = 'get'
		def url(self) -> str:
			return 'https://crosswordnexus.com/finder.php?clue={}'.format(self.url_clue())
		async def get_scores(self) -> Dict[str, float]:
			html = await self.fetch()
			answer_scores = {} # type: Dict[str, float]
			star_tag = r'<img src="/images/star.png" />'
			regex = r'<tr[^>]*>\s*<td[^>]*>\s*((?:{})*)\s*</td>\s*<td[^>]*>\s*<big>\s*<a[^>]*>([\w\s]+)</a>'.format(star_tag)
			matches = re.findall(regex, html)
			for match in matches:
				star_tags, answer = match
				stars = len(star_tags) // len(star_tag)
				value = 1 if stars >= 3 else 0
				answer_scores[answer] = value
			return answer_scores

	class CROSSWORDTRACKER(TrackerBase):
		method = 'get'
		def url(self) -> str:
			return 'http://crosswordtracker.com/search?clue={}'.format(self.url_clue(dash=True))
		async def get_scores(self) -> Dict[str, float]:
			html = await self.fetch()
			answer_scores = {} # type: Dict[str, float]
			regex = r'<li class="answer" data-count="(\d+)" data-text="([\w\s]+)">'
			matches = re.findall(regex, html)
			for match in matches:
				count, answer = match
				answer_scores[answer] = 1
			return answer_scores

	class GOOGLE(TrackerBase):
		method = 'get'
		def url(self) -> str:
			return 'https://www.google.com/search?q={}'.format(self.url_clue())
		async def get_scores(self) -> Dict[str, float]:
			html = await self.fetch()
			answer_scores = {} # type: Dict[str, float]
			wikipeidia_regex = r'>([^<]*) - Wikipedia<'
			matches = re.findall(wikipeidia_regex, html)
			for match in matches:
				answer_scores[match.upper()] = 1
			return answer_scores


if __name__ == '__main__':
	headers = {
		'User-Agent': 'queso',
	}
	session = aiohttp.ClientSession(headers=headers)
	clue = 'batman sidekick'
	# queries = Tracker.get_scores(clue, session)
	# queries2 = Tracker.get_scores('second clue', session)
	q = []
	for i in range(1):
		q.append(Tracker.get_scores_by_tracker('batman sidekick', session))
	all_tasks = asyncio.gather(*q)
	loop = asyncio.get_event_loop()
	answer_scores = loop.run_until_complete(all_tasks)
	loop.run_until_complete(session.close())
	loop.close()
	print(answer_scores)
