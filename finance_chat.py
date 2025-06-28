import requests


def send_chat_message(collection_name: str, message: str):

    url = "http://127.0.0.1:8000/fin_chat"
    params = {
        "collection_name": collection_name,
        "message": message
    }
    response = requests.post(url, params=params)

    if response.status_code == 200:
        print("✅ Response:")
        print(response.json())
    else:
        print("❌ Failed with status code:", response.status_code)
        print(response.text)


def main():
    session_id = "finance_data"  # Replace with actual value
    message = " Controls and Procedures"
    send_chat_message(session_id, message)


if __name__ == "__main__":
    main()
