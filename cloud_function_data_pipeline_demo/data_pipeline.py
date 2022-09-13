# install packages
import os
import json
import logging
import requests
import pandas as pd
import mysql.connector
import pymysql.cursors

# setting database config
DB_CONFIG_REF = {
    'A': A_DB_CONFIG ,
    'B': B_DB_CONFIG,
    'C': C_DB_CONFIG,
    'D': D_DB_CONFIG,
}


def get_data_from_db(query, database='A'):
    """Fetches rows from table(s) of database.
    Retriev rows from table of database instance with specific `query`.
    `config` is the informations of the database instance.
    Args:
        query (str) : sql-like query.
        database (str, optional): the name of the database to connect. Defaults to 'A'.
    Returns:
        pandas.DataFrame: the rows which fit the `query`.
    """
    config = DB_CONFIG_REF[database]
    connection_result = mysql.connector.connect(**config)
    if connection_result:
        return pd.read_sql(query, connection_result)


def get_user_information(user_id: int) -> dict:
    """ using database query user information data
    Args:
        user_id (int): identify users id

    Returns:
       user_info_dict (dict): user information dictionary
    """

    sql_query_str = f"SQL_query"
    try:
        user_info_dataframe = get_data_from_db(query=sql_query_str, database='A')
    except Exception as e:
        print(e)
    user_info_dict = user_info_dataframe.to_dict('records')

    return user_info_dict[0]

# Import list to Saas platform
def import_data(list_id: list, total_string: str)-> dict:

    configuration = v3_sdk.Configuration()
    configuration.api_key['api-key'] = ['APIKEY']

    api_instance = v3_sdk.ContactsApi(v3_sdk.ApiClient(configuration))
    request_import = v3_sdk.RequestContactImport()
    request_import.file_body = total_string
    request_import.list_ids = [list_id]

    try:
        api_response = api_instance.import_(request_import)
        print(f"Import to list {list_id} finished")
    except Exception as e:
        print(f"Exception when calling import: {e}")


def send_email(user_email: str, bcc_email:str, template_id: int):
    configuration = v3_sdk.Configuration()
    configuration.api_key['api-key'] = ['APIKEY']

    api_instance = v3_sdk.TransactionalApi(v3_sdk.ApiClient(configuration))
    email_address_list = [{}]
    email_address_list[0]["email"] = user_email
    bcc_email_address_list = [{}]
    bcc_email_address_list[0]["email"] = bcc_email
    send_email = v3_sdk.SendEmail(to= email_address_list, bcc=bcc_email_address_list, template_id= template_id)

    try:
        api_response = api_instance.send__email(send_email)
    except Exception as e:
        print(f"Exception Transactional: {e}")


def update_data_to_A_saas(order_data, user_data):
    """
    """

    my_headers = {
        'Content-Type': "application/json",
        'Authorization': f"Bearer xxxxxxxxxx-ACCESS_TOKEN']}"
    }

    payload = json.dumps({
    "filterGroups": [
    {
        "filters": [
                {
                    "property_a": "TT",
                    "operator": "EQ",
                    "value": user_data["ccc"]
                }
            ]
        }
    ],
    "properties": [
        "property_a","property_b"
    ],
    "limit": 1,
    "after": 0
    })

    request_response = requests.post(f'https://api.api.com/v3/objects', headers=my_headers, data=payload)

    item_number = len(order_data['data']["products"])
    if item_number == 1:
        o_item = order_data['data']["products"][0]["name"]
    else:
        o_item = ''
        for item in range(item_number):
            if item != item_number-1:
                r_item += f'【{r_data["data"]["p"][item]["name"]}】 & '
            else:
                r_item += f'【{order_data["data"]["p"][item]["name"]}】'

    update_properties = json.dumps({"properties":{
        "AA_key": user_data["AA_value"],
        "AB_key": user_data["AB_value"],
        "AC_key": user_data["AC_value"],
    }})

    response_j = request_response.json()
    update_response = requests.patch(f'https://api.api.com/v3/objects/{response_j["results"][0]["id"]}', headers=my_headers, data=update_properties)

def purchase_success_event(event, context):
    """Background Cloud Function to be triggered by Pub/Sub.
    Args:
         event (dict):  The dictionary with data specific to this type of
                        event. The `@type` field maps to
                         `type.googleapis.com/google.pubsub.v1.PubsubMessage`.
                        The `data` field maps to the PubsubMessage data
                        in a base64-encoded string. The `attributes` field maps
                        to the PubsubMessage attributes if any is present.
         context (google.cloud.functions.Context): Metadata of triggering event
                        including `event_id` which maps to the PubsubMessage
                        messageId, `timestamp` which maps to the PubsubMessage
                        publishTime, `event_type` which maps to
                        `google.pubsub.topic.publish`, and `resource` which is
                        a dictionary that describes the service API endpoint
                        pubsub.googleapis.com, the triggering topic's name, and
                        the triggering event type
                        `type.googleapis.com/google.pubsub.v1.PubsubMessage`.
    Returns:
        None. The output is written to Cloud Logging.
    """
    import base64

    logging.info(
        f'Function was triggered by messageId {context.event_id} published at {context.timestamp} to {context.resource["name"]}')

    raw_data = base64.b64decode(event['data'])
    topic_raw_data = json.loads(raw_data)

    # purchased data
    try:
        ot_id = topic_raw_data['data'][0]["order_transaction_id"]

        headers = {
            'content-type': 'application/json',
            'version': '0.0.0.0',
            'Accept': 'application/json',
            'Authorization': 'Bearer xxxxxxxxxxxxx-xxx-xxxxxxx-xxxxxx'}
        domain = 'https://api.domain.com'

        res = requests.get(
            f'{domain}/api/order/{ot_id}', headers=headers)
        order_data = res.json()

        try:
            data_dict = {}
            data_dict['a_key'] = order_data['M']["P"]
            data_dict['b_key'] = order_data['M']['I']
            data_dict['c_key'] = order_data['M']['T']

            number = len(order_data['data']["p"])
            user_information_dict = get_user_information(
                user_id=order_data['data']["O"])

            if number == 1:
                data_dict['d_key'] = order_data['M']["P"][0]["N"]
            else:
                item = ''
                for item in range(number):
                    if item != number-1:
                        item += f'【{order_data["M"]["P"][item]["name"]}】 & '
                    else:
                        item += f'【{order_data["M"]["P"][item]["name"]}】'
                data_dict['e_key'] = item

            if order_data['O']['S'] == 'AA':
                data_dict['f_kay'] = user_information_dict["name"]
                data_dict['g_key'] = user_information_dict["email"]
            else:
                data_dict['f_kay'] = order_data['M']['P']['n']
                data_dict['f_kay'] = order_data['M']['P']['e']

            import_data(99999, s_data)

            try:
                update_data_to_A_saas(order_data = order_data, user_data = user_information_dict)
            except:
                print('update data to A saas error')

            # send purchased successful information to users by email via A Saas
            if order_data['']['shop'] == 'X':
                send_email(user_email=data_dict['email'], bcc_email='one@domain.com', template_id=100001)
            elif order_data['data']['shop'] == 'Y':
                send_email(user_email=data_dict['email'], bcc_email='one@domain.com', template_id=100002)
        except KeyError:
            print(f'---Incorrect id---')

    except KeyError:
        pass