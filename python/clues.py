import asyncio
from collections import defaultdict
import enum
import functools
import re
import urllib.parse

import aiohttp
from typing import cast, Any, Awaitable, Dict, Optional
from typing_extensions import Literal

'''
Scoring ported from crows project at https://github.com/kcaze/crows.
'''


def urlformat(text : str, dash=False) -> str:
	if dash:
		text = re.sub('\s', '-', text)
	return urllib.parse.quote(text)


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


make_func_value = functools.partial
class Tracker(enum.Enum):
	def __call__(self, *args, **kwargs):
		return self.value(*args, **kwargs)

	@classmethod
	async def get_scores_by_tracker(cls, clue : str, session : aiohttp.ClientSession) -> Dict[str, Dict[str, float]]:
		scores = {
			tracker.name: tracker(clue, session)
			for tracker in cls
		}
		tracker_scores = await gather_resolve_dict(scores)
		return tracker_scores

	@classmethod
	async def get_scores(cls, clue : str, session : aiohttp.ClientSession) -> Dict[str, float]:
		'''Aggregate trackers and reduce by max.'''
		tracker_scores = await cls.get_scores_by_tracker(clue, session)
		scores = defaultdict(float) # type: Dict[str, float]
		for _scores in tracker_scores.values():
			for key, value in _scores.items():
				scores[key] = max(scores[key], value)
		return scores


	@make_func_value
	async def THECROSSWORDSOLVER(clue : str, session : aiohttp.ClientSession) -> Dict[str, float]: # type: ignore # mypy fails to handle decorators correctly
		url = 'http://www.the-crossword-solver.com/search'
		formdata = aiohttp.FormData()
		formdata.add_field('q', clue)
		async with session.post(url, data=formdata) as response:
			html = await response.text()
		answer_scores = {} # type: Dict[str, float]
		regex = r'<p class="searchresult"><a[^>]*>([\w\s]+)</a><span class="matchtypes">(?:<span[^>]*>[^<]*</span>\s*)*</span>'
		matches = re.findall(regex, html)
		for match in matches:
			answer_scores[match] = 1
		return answer_scores

	@make_func_value
	async def WORDPLAYS(clue : str, session : aiohttp.ClientSession) -> Dict[str, float]: # type: ignore # mypy fails to handle decorators correctly
		# requires user agent to be set to non-default
		url = 'https://www.wordplays.com/crossword-solver/{}'.format(urlformat(clue, dash=True))
		async with session.get(url) as response:
			html = await response.text()
		answer_scores = {} # type: Dict[str, float]
		star_tag = r'<div></div>'
		regex = r'<tr[^>]*><td><div class=stars>((?:{})*)</div><div class=clear></div></td><td><a[^>]*>([\w\s]+)</a>'.format(star_tag)
		matches = re.findall(regex, html)
		for match in matches:
			star_tags, answer = match
			stars = len(star_tags) // len(star_tag)
			value = 1 if stars >= 4 else 0
			answer_scores[answer] = value
		return answer_scores

	@make_func_value
	async def CROSSWORDNEXUS(clue : str, session : aiohttp.ClientSession) -> Dict[str, float]: # type: ignore # mypy fails to handle decorators correctly
		url = 'https://crosswordnexus.com/finder.php?clue={}'.format(urlformat(clue))
		async with session.get(url) as response:
			html = await response.text()
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

	@make_func_value
	async def CROSSWORDTRACKER(clue : str, session : aiohttp.ClientSession) -> Dict[str, float]: # type: ignore # mypy fails to handle decorators correctly
		url = 'http://crosswordtracker.com/search?clue={}'.format(urlformat(clue, dash=True))
		async with session.get(url) as response:
			html = await response.text()
		answer_scores = {} # type: Dict[str, float]
		regex = r'<li class="answer" data-count="(\d+)" data-text="([\w\s]+)">'
		matches = re.findall(regex, html)
		for match in matches:
			count, answer = match
			answer_scores[answer] = 1
		return answer_scores

	@make_func_value
	async def GOOGLE(clue : str, session : aiohttp.ClientSession) -> Dict[str, float]: # type: ignore # mypy fails to handle decorators correctly
		url = 'https://www.google.com/search?q={}'.format(urlformat(clue))
		async with session.get(url) as response:
			html = await response.text()
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
	for i in range(10):
		q.append(Tracker.get_scores_by_tracker('batman sidekick', session))
	all_tasks = asyncio.gather(*q)
	loop = asyncio.get_event_loop()
	answer_scores = loop.run_until_complete(all_tasks)
	loop.run_until_complete(session.close())
	loop.close()
	print(answer_scores)
