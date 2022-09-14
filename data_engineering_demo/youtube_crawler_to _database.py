import abc
import datetime
import json
import os
from urllib.parse import quote, urlparse

from flask import request
from google.cloud import datastore, pubsub_v1
from selenium import webdriver

from ..common.util import (background_processing, get_data_from_bq,
                           get_data_from_db, logger, worker)

AUTO_INSERT_VIDEOS_JOBS_LIMIT = 250


class CrawlerInterface(abc.ABC):
    @abc.abstractmethod
    def click(self):
        pass

    @abc.abstractmethod
    def scroll(self):
        pass

    @abc.abstractmethod
    def fetch(self):
        pass


class Crawler(CrawlerInterface):
    def __init__(self, urls, max_scroll, uploader_id=None):
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-gpu')
        self.browser = webdriver.Chrome(options=chrome_options)
        self.urls = urls
        self.max_scroll = max_scroll
        self.uploader_id = uploader_id
        self.publisher = pubsub_v1.PublisherClient()
        self.ids = set()
        self.videos_count = 0

    def click(self, url):
        self.browser.get(url)
        self.browser.implicitly_wait(1)

    def scroll(self):
        js = f'window.scrollTo(0, document.documentElement.scrollHeight)'
        self.browser.execute_script(js)

    def fetch(self):
        elements = self.browser.find_elements_by_xpath(
            "//ytd-thumbnail/a[@id='thumbnail']//yt-img-shadow/img[@src]")
        video_prefix = 'i.ytimg.com'
        # the channel prefix is 'yt3.ggpht.com'
        for element in elements:
            src = element.get_attribute('src')
            if src is not None:
                parsed = urlparse(src)
                if parsed.hostname == video_prefix:
                    id_ = parsed.path.split('/')[2]
                    self.ids.add(id_)

    # future deprecated
    def is_over_length(self, video_length):
        threshold = 1200
        splited_video_length = video_length.split(':')
        # for mm:ss format
        if len(splited_video_length) == 2:
            _min, sec = splited_video_length
            return 60 * int(_min) + int(sec) >= threshold
        # for hh:mm:ss format
        elif len(splited_video_length) == 3:
            hr, _min, sec = splited_video_length
            return 3600 * int(hr) + 60 * int(_min) + int(sec) >= threshold

    def publish(self, data, topic):
        data = json.dumps(data).encode('utf-8')
        topic_path = self.publisher.topic_path('PROJECT', topic)
        future = self.publisher.publish(topic_path, data=data)
        future.result()

    def remove_imported_videos(self):
        cmd = 'sql query'
        self.ids = self.ids - set(get_data_from_db(cmd).youtube_id)

    def remove_errored_transcribed_videos(self):
        current_date = datetime.date.today()
        delta = datetime.timedelta(90)
        # currently remove videos with no captions after trabcribed
        cmd = f"sql query"
        self.ids = self.ids - set(get_data_from_bq(cmd).youtube_id)

    def get_id(self):
        return self.ids

    def terminate(self):
        self.browser.close()

    @background_processing
    def action(self):
        for index, url in enumerate(self.urls):
            if self.videos_count >= AUTO_INSERT_VIDEOS_JOBS_LIMIT:
                break

            self.click(url)
            for _ in range(self.max_scroll):
                self.fetch()
                self.scroll()
            self.fetch()

            if len(self.ids) >= 100 or index == len(self.urls) - 1:
                # Batchly processing crawled videos,
                # publish to pub/sub after filter
                self.remove_imported_videos()
                self.remove_errored_transcribed_videos()
                youtube_ids = self.get_id()
                for youtube_id in youtube_ids:
                    data = {'youtubeId': youtube_id}
                    if self.uploader_id:
                        data['uploaderId'] = self.uploader_id
                    self.publish(data, 'autoInsertVideos')
                self.videos_count += len(youtube_ids)
                self.ids = set()
        self.terminate()
        message = f'> Crawl success\n> There are {self.videos_count} videos in Total.'
        logger.info(message)


def worker_with_limited_queue(func, max_worker, **arg):
    if worker._work_queue.qsize() <= max_worker:
        worker.submit(func, **arg)


def crawler_handler() -> dict:
    """path receive `/{{version}}/crawler/{{crawler_category}}`"""

    crawler_category = request.path.split('/')[3]
    max_scroll = int(request.args.get('scroll', 2))
    uploader_id = int(request.args.get('uploaderId'))

    urls: str = None
    # receive POST method
    if crawler_category == 'urls':
        req_body = request.get_json(force=True)
        urls = [url for url in req_body['urls']]

    # receive POST method
    elif crawler_category == 'keywords':
        req_body = request.get_json(force=True)
        urls = [
            f'https://www.youtube.com/results?search_query={quote(keyword)}' for keyword in req_body['keywords']]

    # receive POST method
    elif crawler_category == 'channels':
        req_body = request.get_json(force=True)
        urls = [
            f'https://www.youtube.com/channel/{channel}/videos' for channel in req_body['channels']]

    # receive GET method
    elif crawler_category == 'scheduled_crawler':
        # query datastore
        kind = 'crawler_channels'
        client = datastore.Client()
        query = client.query(kind=kind)
        query.add_filter("uploader_id", "=", uploader_id)
        results = list(query.fetch())

        urls_channels = [
            f'https://www.youtube.com/channel/{d["channel_id"]}/videos' for d in results if d.get("channel_id")]
        urls_alias = [
            f'https://www.youtube.com/user/{d["channel_alias"]}/videos' for d in results if d.get("channel_alias")]
        urls = urls_channels + urls_alias

    if not urls:
        return {"data": "invalid `crawler_category`"}

    crawler = Crawler(urls, max_scroll, uploader_id)
    crawler.action()

    return {"data": "ok"}
