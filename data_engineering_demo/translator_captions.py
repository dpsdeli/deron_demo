import base64
import copy
import json
import os
import tempfile
from datetime import datetime
from typing import Tuple
import time

import jieba
import opencc
import requests
from flask import request
from flask_restful.inputs import boolean
from google.cloud.storage import Client

from ..common.cross_trait import cross_trait_ref
from ..common.util import (get_data_from_db, retry_get_data_from_db, insert_data_to_db, logger,
                           publish, verify_google_oauth_token, worker, send_to_slack)

AUTO_TRANSLATION_WORKAROUND_WAITING_SECONDS = 3


class CrossTrait:
    ref = cross_trait_ref

    def convert(self, text):
        tokens = jieba.cut(text)
        ref = CrossTrait.ref
        data = []
        for token in tokens:
            if token in ref:
                data.append(ref[token])
            else:
                data.append(token)
        return ''.join(data)


def _simplified_to_traditional(simplified):
    converter = opencc.OpenCC('s2tw.json')
    return converter.convert(simplified)


def _deepl(text, target_lang):
    language_reference_table = {
        "zh-TW": "zh",
        "zh-CN": "zh",
        "ja-JP": "ja",
    }
    target_lang_for_deepl = target_lang

    if target_lang in language_reference_table:
        target_lang_for_deepl = language_reference_table[target_lang]
    url = 'https://api.translator.com/'
    body = {
        "auth_key": os.environ['API_KEY'],
        "text": text,
        "source_lang": "EN",
        "target_lang": target_lang_for_deepl,
    }
    response = requests.post(url, data=body)

    if response.status_code != 200:
        raise ValueError("translate service fails")

    l_data = json.loads(response.text)
    translated = []

    for index, item in enumerate(l_data['translations']):
        translation = item['text']
        translated_caption_details = translation
        item['original_text'] = text[index]
        translated.append({
            "translated_text": translated_caption_details,
            "service_response": json.dumps(item, ensure_ascii=False)
        })

    return translated


def check_caption(video_id: int, target_lang: str) -> bool:
    statement = f"""sql string"""
    df = get_data_from_db(statement)
    return df.empty


def simplified_to_traditional(translation: str) -> str:
    cross_trait_text = CrossTrait().convert(translation)
    traditional_translation = _simplified_to_traditional(cross_trait_text)
    return traditional_translation


def switch_cn_tw(cn_or_tw: str) -> str:
    ref = {
        "zh-TW": "zh-CN",
        "zh-CN": "zh-TW",
    }
    return ref[cn_or_tw]


def caption_details_worker(
    video_id: int,
    target_lang: str
) -> bool:
    caption_details_language_ref = {
        "zh-TW": "zh-Hant",
        "zh-CN": "zh-Hans",
        "ja-JP": "ja",
    }
    caption_lang = target_lang

    if caption_lang in caption_details_language_ref:
        caption_lang = caption_details_language_ref[caption_lang]

    sql_query_ = f'''
        sql_string
    '''

    insert_stmt = f'''
        sql_string
    '''

    query_time = time.time()
    translation_df = retry_get_data_from_db(sql_query_)

    if translation_df.empty:
        message = f'`translation`: video id {video_id} already translated at query time: {query_time}'
        logger.info(message)
        return False

    CAPTION_UPPER_BOUND = 500
    if len(translation_df) > CAPTION_UPPER_BOUND:
        # NOTE inhibit caption translation when captions more than `CAPTION_UPPER_BOUND`
        message = f'`translation`: video id `{video_id}` has caption details more than {CAPTION_UPPER_BOUND}.'
        logger.info(message)
        return False

    # numpy int64 to python int64
    caption_id = int(translation_df.caption_id.unique()[0])
    texts = translation_df.original_text.to_list()
    caption_detail_ids = translation_df.caption_detail_id.to_list()
    # check if texts and caption_detail_ids has the equal length
    loops = list(zip(texts, caption_detail_ids))
    batch = 40

    for step in range(0, len(loops), batch):
        translated = _dl(texts[step:step + batch], target_lang)
        translated_tw = []
        for index, item in enumerate(translated):
            item['translated_caption_details'] = item.pop('translated_text')
            item.update({
                "video_id": video_id,
                "caption_id": caption_id,
                "caption_detail_id": caption_detail_ids[step + index],
                "target_lang": target_lang
            })

            if 'zh' in target_lang:
                item['target_lang'] = 'zh-CN'

                item_zh_tw = copy.deepcopy(item)
                item_zh_tw['target_lang'] = 'zh-TW'
                item_zh_tw['translated_caption_details'] = simplified_to_traditional(
                    item_zh_tw['translated_caption_details'])
                translated_tw.append(item_zh_tw)

        insert_data_to_db(insert_stmt, [*translated, *translated_tw])

    return True


def translate_title_worker(video_id: int, target_lang: str) -> bool:
    title_language_ref = {
        "zh-TW": "zh_tw",
        "zh-CN": "zh_tw",
        "ja-JP": "ja",
    }
    title_lang = target_lang

    if title_lang in title_language_ref:
        title_lang = title_language_ref[title_lang]

    sql_title_string = f'''sql query'''
    sql_insert_string = f'''sql query'''

    query_time = time.time()
    title_df = retry_get_data_from_db(sql_title_string)

    if title_df.empty:
        message = f'sql query'
        logger.info(message)
        return False

    texts = title_df.original_text.to_list()[:1]
    translated_array = trans_(texts, target_lang)
    translated = translated_array[0]
    translated_text = translated.pop('translated_text')

    if texts[0] == '【' and texts[0] != translated_text[0]:
        # NOTE work around for automaticly trimming first non-alphabetic character
        translated_text = '【' + translated_text

    translated['translated_title'] = translated_text
    translated.update({
        "video_id": video_id,
        "target_lang": target_lang
    })

    if 'zh' in target_lang:
        translated['target_lang'] = 'zh-CN'

        translated_tw = copy.copy(translated)
        translated_tw['target_lang'] = 'zh-TW'
        translated_tw['translated_title'] = simplified_to_traditional(
            translated_tw['translated_title'])
        translated_array.append(translated_tw)

    insert_data_to_db(sql_insert_string, translated_array)

    return True


def translator_worker(
    video_id: int,
    target_lang: str,
    is_title: bool = True,
    is_caption: bool = True
):

    try:
        is_title_translated = False
        is_caption_translated = False

        if not is_title and not is_caption:
            logger.info("Both `is_title` and `is_caption` are false.")
            return

        if is_title:
            is_title_translated = translate_title_worker(video_id, target_lang)

        if is_caption:
            if check_caption(video_id, target_lang):
                is_caption_translated = caption_details_worker(
                    video_id, target_lang)
            else:
                message = f'`translation`: video id `{video_id}` already has {target_lang} caption detail.'
                logger.info(message)

        if (is_title and is_title_translated) or (is_caption and is_caption_translated):
            pub_sub_data = {
                "videoId": video_id,
                "targetLanguage": target_lang,
                "isTitle": is_title and is_title_translated,
                "isCaption": is_caption and is_caption_translated
            }
            publish(pub_sub_data, 'updateTranslationResult')

            if 'zh' in target_lang:
                pub_sub_data['targetLanguage'] = switch_cn_tw(target_lang)
                publish(pub_sub_data, 'updateTranslationResult')

    except Exception as e:
        logger.exception(str(e))

    send_to_slack(
        f"ranslate job of video_id:[{video_id}] for '{target_lang}' is done!", channel="notification")


@verify_google_oauth_token
def video_translator_pub_sub() -> Tuple[str, int]:
    '''unpacked format should be
        {
            "videoId": {{id}},
            "targetLanguage": {{zh-TW | zh-CN | ja-JP}},
        }
    '''
    req = request.get_json(force=True)
    data = req.get('message', {}).get('data')
    if data:
        video_id = json.loads(base64.b64decode(data).decode()).get('videoId')
        target_lang = json.loads(base64.b64decode(
            data).decode()).get('targetLanguage')

        if not video_id or not target_lang:
            return 'Format is not correct', 200

        if worker._work_queue.qsize() <= 1:
            worker.submit(translator_worker, video_id, target_lang)
            return 'Processing', 200

        return 'Too Many Requests', 429

    return 'No Content', 204


def video_translator() -> dict:
    video_id = request.args.get('videoId', 0)
    target_lang = request.args.get('targetLanguage', '')
    translator_worker(video_id, target_lang)
    send_to_slack(
        f"Translate video captions job of video_id:[{video_id}] is done!")

    return {"data": "ok"}


def video_title_checker():
    date = request.args.get("date")
    if not date:
        date = datetime.today().strftime("%Y-%M-%d")

    check_df = get_data_from_db(f"""sql string""")
    with tempfile.NamedTemporaryFile() as temp:
        check_df.to_csv(temp.name + '.csv')
        storage_client = Client()
        bucket = storage_client.get_bucket('STORAGE')

        blob = bucket.blob(f'./check_{int(time.time())}.csv')
        blob.upload_from_file(temp)

    return 'OK'
