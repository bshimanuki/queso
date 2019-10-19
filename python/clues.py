import asyncio
import re
import urllib.parse

import aiohttp
from typing import Dict

'''
Scoring ported from crows project at https://github.com/kcaze/crows.
'''


def urlformat(text : str, dash=False) -> str:
	if dash:
		text = re.sub('\s', '-', text)
	return urllib.parse.quote(text)

async def fetch_thecrosswordsolver(clue : str, session : aiohttp.ClientSession) -> Dict[str, float]:
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

async def fetch_wordplays(clue : str, session : aiohttp.ClientSession) -> Dict[str, float]:
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

async def fetch_crosswordnexus(clue : str, session : aiohttp.ClientSession) -> Dict[str, float]:
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

async def fetch_crosswordtracker(clue : str, session : aiohttp.ClientSession) -> Dict[str, float]:
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

async def fetch_google(clue : str, session : aiohttp.ClientSession) -> Dict[str, float]:
	url = 'https://www.google.com/search?q={}'.format(urlformat(clue))
	async with session.get(url) as response:
		html = await response.text()
	answer_scores = {} # type: Dict[str, float]
	wikipeidia_regex = r'>([^<]*) - Wikipedia<'
	matches = re.findall(wikipeidia_regex, html)
	for match in matches:
		answer_scores[match.upper()] = 1
	return answer_scores

trackers = [fetch_thecrosswordsolver, fetch_wordplays, fetch_crosswordnexus, fetch_crosswordtracker, fetch_google]


if __name__ == '__main__':
	# test
	# async with aiohttp.ClientSession() as session:
	headers = {
		'User-Agent': 'queso',
	}
	session = aiohttp.ClientSession(headers=headers)
	loop = asyncio.get_event_loop()
	clue = 'batman sidekick'
	queries = [fetch(clue, session) for fetch in trackers]
	futures = asyncio.gather(*queries)
	answer_scores = loop.run_until_complete(futures)
	loop.close()
	print(answer_scores)
