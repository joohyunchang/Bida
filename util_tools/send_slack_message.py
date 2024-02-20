import requests
import sys

def send_slack_message(token, channel, message):
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    payload = {
        'channel': channel,
        'text': message
    }
    response = requests.post('https://slack.com/api/chat.postMessage', headers=headers, json=payload)
    if response.status_code != 200:
        raise ValueError(f"Request to slack returned an error {response.status_code}, the response is:\n{response.text}")

if __name__ == "__main__":
    token = sys.argv[1]
    channel = sys.argv[2]  # 채널 ID
    message = sys.argv[3]
    send_slack_message(token, channel, message)
