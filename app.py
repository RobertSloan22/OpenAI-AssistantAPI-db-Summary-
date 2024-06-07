import os
import time
import requests
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI, AssistantEventHandler
from langchain_community.utilities import SQLDatabase
from tinydb import TinyDB, Query
from discord.ext import commands, tasks
from datetime import datetime, timedelta, timezone
import asyncio
import discord
import psycopg2

load_dotenv()

# Use .env file for API keys
REAL_API_KEY = os.getenv("REAL_OPENAI_API_KEY")
FAKE_API_KEY = os.getenv("FAKE_OPENAI_API_KEY")
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
DB_CONN_DETAILS = os.getenv("DB_CONN_DETAILS", "dbname='' user='' host='' password=''")

# Corrected connection string for SQLAlchemy
DB_CONN_DETAILS_SQLALCHEMY = ""

# Database setup
db = SQLDatabase.from_uri(DB_CONN_DETAILS_SQLALCHEMY)

# Set up TinyDB for storing Discord messages
db_tiny = TinyDB("discord_messages.json")

# Function to switch between fake and real API
def switch_api(use_fake):
    global client
    api_key = FAKE_API_KEY if use_fake else REAL_API_KEY
    client = OpenAI(api_key=api_key)

# Switch API based on sidebar selection
use_fake_server = st.sidebar.radio("Select API Server", ["Real OpenAI API", "Fake OpenAI API"]) == "Fake OpenAI API"
switch_api(use_fake_server)

# OpenAI setup
client = OpenAI(api_key=REAL_API_KEY)

# Discord headers
headers = {
    "Authorization": os.getenv("AUTH_TOKEN"),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Sec-CH-UA": "\"Not A(Brand\";v=\"99\", \"Microsoft Edge\";v=\"121\"; \"Chromium\";v=\"121\"",
    "Sec-CH-UA-Mobile": "?0",
    "Sec-CH-UA-Platform": "\"Windows\"",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "X-Debug-Options": "bugReporterEnabled",
    "X-Discord-Locale": "en-US",
    "X-Discord-Timezone": "America/Los_Angeles",
}

# Discord channels
channels = [
    {"id": "1132706834340397116", "nickname": "Winners group chat"},
    {"id": "1106390097378684983", "nickname": "TRAC: main-chat"},
    {"id": "1157429728282673264", "nickname": "TRAC: pipe"},
    {"id": "1138509209525297213", "nickname": "TRAC: tap-protocol"},
    {"id": "1236020558488141874", "nickname": "TRAC: gib"},
    {"id": "1115824966470991923", "nickname": "OnlyFarmers: alpha"},
    {"id": "1166459733075579051", "nickname": "Ordicord: ordinals coding club 4/10/2024"},
    {"id": "1224564960575623269", "nickname": "Taproot Alpha: runes"},
    {"id": "1084525778852651190", "nickname": "DogePunks: holder-chat"},
    {"id": "1010230594367655996", "nickname": "Tensor: alpha"},
    {"id": "987504378749538366", "nickname": "Ordicord: general"},
    {"id": "1069465367988142110", "nickname": "Ordicord: tech-support"},
]

# Function to fetch Discord messages
def fetch_discord_messages(channel_id):
    url = f"https://discord.com/api/v9/channels/{channel_id}/messages?limit=100"
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch messages for channel {channel_id}: HTTP {response.status_code} - {response.text}")
            return []
    except requests.exceptions.RequestException as e:
        st.error(f"Request exception for channel {channel_id}: {e}")
        return []

# Function to insert new messages into TinyDB
def insert_new_messages(messages, channel_id, nickname):
    for message in messages:
        Message = Query()
        if not db_tiny.search(Message.id == message["id"]):
            referenced_message = message.get("referenced_message", {})
            if referenced_message is None:
                referenced_message = {}
            attachments = message.get("attachments", [])
            attachment_file_name = None
            attachment_url = None

            if attachments:
                attachment_file_name = attachments[0].get("filename")
                attachment_url = attachments[0].get("url")

            db_tiny.insert({
                "id": message["id"],
                "channel_id": channel_id,
                "nickname": nickname,
                "content": message.get("content"),
                "timestamp": message.get("timestamp"),
                "author_id": message["author"]["id"],
                "author_username": message["author"]["username"],
                "author_global_name": message["author"].get("global_name"),
                "referenced_message_id": referenced_message.get("id"),
                "referenced_message_content": referenced_message.get("content"),
                "referenced_message_username": referenced_message.get("author", {}).get("username"),
                "referenced_message_global_name": referenced_message.get("author", {}).get("global_name"),
                "attachment_file_name": attachment_file_name,
                "attachment_url": attachment_url,
            })

# Function to fetch messages with references from the database
def fetch_messages_with_references(channel_id, since):
    with psycopg2.connect(DB_CONN_DETAILS) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT m.message_id, m.content, m.author_username, m.timestamp, m.referenced_message_id, ref.message_id AS ref_message_id, ref.content AS ref_content, ref.author_username AS ref_author_username
                FROM discord_messages m
                LEFT JOIN discord_messages ref ON m.referenced_message_id = ref.message_id
                WHERE m.channel_id = %s AND m.timestamp > %s
                ORDER BY m.timestamp ASC
            """, (channel_id, since))
            messages = cur.fetchall()
    return messages
def run_assistant():
    client = OpenAI(api_key=REAL_API_KEY)

    assistant = client.beta.assistants.create(
        name="Data Analyzer Assistant",
        instructions="You are a helpful assistant. Focus on the most recent 200 messages acting as an alpha caller to summarize important information from the messages, including any replies to referenced messages by the referenced user. The messages include content, author username, timestamp, and if applicable, referenced message content and referenced message author username. Do not mention the user names specifically; we want to keep them private. Summarize in bullet points in a medium (discord message) format. Disregard any information that is just chatter. Be very detailed if the alpha or information is important. Keep it so simple that even someone with ADD can read and follow it. Provide a concise summary of the content. Always include links in each statement to the original messages for further context so users can reference the message the convo it is referencing.Go Through all the user comments, taking note of the conversation topics, specific mentions of runes, crypto currency's, if something is mooning, who said what and when, Reporting back with a highly detailed bulleted list of all accounts. The bulleted list needs to contain  the username the comment, the time and date, the topic and the channel. Pay very close attention to the Winners channel and the Alpha channels.  Aggregate the Bulleted lists by channel id. Give the most importance and attention to the comments that have happened starting most recently and working backwards. As more time elapses from the point of the comment its value decreases. All comments in the past Week maintain the top level importance.  DO NOT DO THE FOLLOWING: DO NOT LIST OUT REFERENCES IN THIS MANNER ([1][2][3][4][5][6][7][8][9][10][11][12][13][14][15][16][17]). All data that is returned to the user must be in the form of a natural language sentence as part of a bulleted list. When reporting the summary and analysis back to the user, ensure that the summary structure Indicates: 'SUMMARY FOR TODAY[TODAYS DATE] '.. then list summary of all channels, conversations, username and comments, with a  paragraph over view. Then move to 'SUMMARY FOR YESTERDAY[YESTERDAYS DATE]' , .. then list summary of all channels, conversations, username and comments, with a  paragraph over view. Then move to  'SUMMARY FOR [DATE]' .. then list summary of all channels, conversations, username and comments, with a  paragraph over view. Working backwards in time one day at a time.",
        model="gpt-4o",
        tools=[{"type": "file_search"}],
    )

    # Use TinyDB file as the data source
    file_paths = ["discord_messages.json"]
    file_streams = [open(path, "rb") for path in file_paths]

    vector_store = client.beta.vector_stores.create(name="Data Files")
    file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vector_store.id, files=file_streams
    )

    vector_store_files = client.beta.vector_stores.files.list(vector_store_id=vector_store.id)
    file_ids = [file.id for file in vector_store_files]

    assistant = client.beta.assistants.update(
        assistant_id=assistant.id,
        tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}}
    )

    thread = client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": "You are a helpful assistant. Focus on the most recent 200 messages acting as an alpha caller to summarize important information from the messages, including any replies to referenced messages by the referenced user. The messages include content, author username, timestamp, and if applicable, referenced message content and referenced message author username. Do not mention the user names specifically; we want to keep them private. Summarize in bullet points in a medium (discord message) format. Disregard any information that is just chatter. Be very detailed if the alpha or information is important. Keep it so simple that even someone with ADD can read and follow it. Provide a concise summary of the content. Always include links in each statement to the original messages for further context so users can reference the message the convo it is referencing.Go Through all the user comments, taking note of the conversation topics, specific mentions of runes, crypto currency's, if something is mooning, who said what and when, Reporting back with a highly detailed bulleted list of all accounts. The bulleted list needs to contain  the username the comment, the time and date, the topic and the channel. Pay very close attention to the Winners channel and the Alpha channels.  Aggregate the Bulleted lists by channel id. Give the most importance and attention to the comments that have happened starting most recently and working backwards. As more time elapses from the point of the comment its value decreases. All comments in the past Week maintain the top level importance.  DO NOT DO THE FOLLOWING: DO NOT LIST OUT REFERENCES IN THIS MANNER ([1][2][3][4][5][6][7][8][9][10][11][12][13][14][15][16][17]). All data that is returned to the user must be in the form of a natural language sentence as part of a bulleted list. When reporting the summary and analysis back to the user, ensure that the summary structure Indicates: 'SUMMARY FOR TODAY[TODAYS DATE] '.. then list summary of all channels, conversations, username and comments, with a  paragraph over view. Then move to 'SUMMARY FOR YESTERDAY[YESTERDAYS DATE]' , .. then list summary of all channels, conversations, username and comments, with a  paragraph over view. Then move to  'SUMMARY FOR [DATE]' .. then list summary of all channels, conversations, username and comments, with a  paragraph over view. Working backwards in time one day at a time.",
                "attachments": [
                    {"file_id": file_id, "tools": [{"type": "file_search"}]} for file_id in file_ids
                ],
            }
        ]
    )

    class EventHandler(AssistantEventHandler):
        def on_text_created(self, text) -> None:
            st.write(f"assistant > {text}")

        def on_tool_call_created(self, tool_call):
            st.write(f"assistant > {tool_call.type}")

        def on_message_done(self, message) -> None:
            message_content = message.content[0].text
            annotations = message_content.annotations
            citations = []
            for index, annotation in enumerate(annotations):
                message_content.value = message_content.value.replace(
                    annotation.text, f"[{index}]"
                )
                if file_citation := getattr(annotation, "file_citation", None):
                    cited_file = client.files.retrieve(file_citation.file_id)
                    citations.append(f"[{index}] {cited_file.filename}")

            st.write(message_content.value)
            st.write("\n".join(citations))

    with client.beta.threads.runs.stream(
        thread_id=thread.id,
        assistant_id=assistant.id,
        instructions="You are a helpful assistant. Focus on the most recent 200 messages acting as an alpha caller to summarize important information from the messages, including any replies to referenced messages by the referenced user. The messages include content, author username, timestamp, and if applicable, referenced message content and referenced message author username. Do not mention the user names specifically; we want to keep them private. Summarize in bullet points in a medium (discord message) format. Disregard any information that is just chatter. Be very detailed if the alpha or information is important. Keep it so simple that even someone with ADD can read and follow it. Provide a concise summary of the content. Always include links in each statement to the original messages for further context so users can reference the message the convo it is referencing.Go Through all the user comments, taking note of the conversation topics, specific mentions of runes, crypto currency's, if something is mooning, who said what and when, Reporting back with a highly detailed bulleted list of all accounts. The bulleted list needs to contain  the username the comment, the time and date, the topic and the channel. Pay very close attention to the Winners channel and the Alpha channels.  Aggregate the Bulleted lists by channel id. Give the most importance and attention to the comments that have happened starting most recently and working backwards. As more time elapses from the point of the comment its value decreases. All comments in the past Week maintain the top level importance.  DO NOT DO THE FOLLOWING: DO NOT LIST OUT REFERENCES IN THIS MANNER ([1][2][3][4][5][6][7][8][9][10][11][12][13][14][15][16][17]). All data that is returned to the user must be in the form of a natural language sentence as part of a bulleted list. When reporting the summary and analysis back to the user, ensure that the summary structure Indicates: 'SUMMARY FOR TODAY[TODAYS DATE] '.. then list summary of all channels, conversations, username and comments, with a  paragraph over view. Then move to 'SUMMARY FOR YESTERDAY[YESTERDAYS DATE]' , .. then list summary of all channels, conversations, username and comments, with a  paragraph over view. Then move to  'SUMMARY FOR [DATE]' .. then list summary of all channels, conversations, username and comments, with a  paragraph over view. Working backwards in time one day at a time.",
        event_handler=EventHandler(),
    ) as stream:
        stream.until_done()

    for stream in file_streams:
        stream.close()


# Function to handle post-summary conversation
def post_summary_conversation(summary):
    st.write("### Summary")
    st.write(summary)

    user_input = st.text_input("Ask a follow-up question about the summary")
    if st.button("Ask"):
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "user", "content": user_input},
                    {"role": "system", "content": summary},
                ],
            )
            answer = response.choices[0].message['content']
            st.write("### Response")
            st.write(answer)
        except Exception as e:
            st.error(f"Error generating response: {e}")

# Discord bot setup
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True  # Enable message content intent
bot = commands.Bot(command_prefix="!", intents=intents)

@tasks.loop(hours=.016)
async def send_summaries():
    channel = bot.get_channel(1220429867288625162)
    if channel is None:
        st.error("Summary channel not found.")
        return

    for channel_id in ['']:
        summary = await generate_summary(channel_id)
        if summary and summary.startswith("Failed"):
            continue  # Skip sending if the summary generation failed
        message_content = f"**Summary for <#{channel_id}> for the last hour:**\n{summary}"
        if len(message_content) > 2000:
            message_content is message_content[:1997] + "..."
        await channel.send(message_content)

async def generate_summary(channel_id):
    since = datetime.now(timezone.utc) - timedelta(hours=1)
    messages = fetch_messages_with_references(channel_id, since)
    if not messages:
        return "No new messages to summarize."

    try:
        with open('prompt.txt', 'r') as file:
            system_prompt = file.read().strip()
    except FileNotFoundError:
        st.error("The file prompt.txt was not found.")
        return "Error: System prompt file not found."

    prompt_messages = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        message_time = msg[3] + timedelta(hours=-8)
        unix_timestamp = int(message_time.timestamp())
        message_link = f"https://discord.com/channels/{os.getenv('GUILD_ID')}/{channel_id}/{msg[0]}"
        message_content = f"{msg[2]} said: {msg[1]} (Sent at <t:{unix_timestamp}:F> [Link]({message_link}))"
        if msg[4]:
            ref_message_link = f"https://discord.com/channels/{os.getenv('GUILD_ID')}/{channel_id}/{msg[4]}"
            message_content += f" [Replying to a message from {msg[7]}]({ref_message_link})"
        prompt_messages.append({"role": "user", "content": message_content})

    for attempt in range(5):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4-0125-preview",
                messages=prompt_messages
            )
            return response.choices[0].message['content']
        except openai.error.APIConnectionError as e:
            st.error(f"OpenAI API connection error on attempt {attempt + 1}: {e}")
            if attempt < 4:
                time.sleep(10)
    return "Failed to generate a summary after multiple attempts due to connection issues."

@bot.event
async def on_ready():
    st.write(f'Logged in as {bot.user.name} ({bot.user.id})')
    send_summaries.start()  # Start the loop
    await send_summaries()  # Run the summary task immediately

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if bot.user.mentioned_in(message):
        prompt = message.content.replace(f'<@!{bot.user.id}>', '').strip()
        asyncio.create_task(handle_mention(message, prompt))

    await bot.process_commands(message)

async def handle_mention(message, prompt):
    retry_attempts = 3
    for attempt in range(retry_attempts):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4-0125-preview",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500
            )
            response_text = response.choices[0].message['content']
            chunks = [response_text[i:i+2000] for i in range(0, len(response_text), 2000)]
            for chunk in chunks:
                await message.channel.send(chunk)
            break  # Exit the loop if the request was successful
        except openai.error.APIConnectionError as e:
            if attempt < retry_attempts - 1:
                await message.channel.send("I'm having trouble connecting to my brain. Trying again...")
                st.error(f"OpenAI API connection error: {e}, attempt {attempt + 1}")
                await asyncio.sleep(2)
            else:
                await message.channel.send("I'm still having trouble after several attempts. Please try again later.")
                st.error(f"Final OpenAI API connection error: {e}")
        except discord.errors.HTTPException as e:
            st.error(f"Discord HTTP exception: {e}")
            break

def run_sorabot():
    if not DISCORD_BOT_TOKEN:
        st.error("Discord bot token not found. Please check your environment variables.")
        return

    bot.run(DISCORD_BOT_TOKEN)

# Streamlit app setup
st.title("Discord Data Analysis Assistant --Version .02")
st.sidebar.title("Options - Updates pending")

# Function to handle commands
command = st.sidebar.selectbox(
    "Choose a command",
    ["Fetch Discord Messages", "Analyze Data", "Post Summary Conversation"]
)

if command == "Fetch Discord Messages":
    channel_selection = st.selectbox(
        "Choose a Discord Channel",
        options=[{"id": channel["id"], "label": f'{channel["nickname"]} (ID: {channel["id"]})'} for channel in channels],
        format_func=lambda option: option["label"]
    )
    if channel_selection:
        channel_id = channel_selection["id"]
        channel_nickname = [channel["nickname"] for channel in channels if channel["id"] == channel_id][0]
        if st.button("Fetch Messages"):
            messages = fetch_discord_messages(channel_id)
            if messages:
                st.write(f"Fetched {len(messages)} messages")
                insert_new_messages(messages, channel_id, channel_nickname)
                run_assistant()  # Automatically run the assistant after fetching messages

if command == "Analyze Data":
    if st.button("Run Assistant"):
        run_assistant()

if command == "Post Summary Conversation":
    summary = st.text_area("Summary of recent messages")
    if st.button("Start Post-Summary Conversation"):
        post_summary_conversation(summary)

if command == "Natural Language Query":
    user_input = st.text_input("Ask a question")
    if st.button("Ask"):
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "user", "content": user_input},
                ],
            )
            answer = response.choices[0].message['content']
            st.write(answer)
        except Exception as e:
            st.error(f"Error generating response: {e}")

if command == "SoraBot":
    st.write("Running SoraBot...")
    run_sorabot()
