#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from datetime import datetime, timedelta
import openai
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import requests
import json
import os
from dateutil.parser import parse, ParserError
import base64
from email.mime.text import MIMEText

def get_credentials(client_secret_file, scopes):
    flow = InstalledAppFlow.from_client_secrets_file(client_secret_file, scopes)
    credentials = flow.run_local_server(port=0)
    return credentials

SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.send',
    'https://www.googleapis.com/auth/calendar',
]

client_secret_file = 'client_secret.json'

# Get credentials for the Gmail account
print("Authorize the Gmail account")
credentials = get_credentials(client_secret_file, SCOPES)

# Save the credentials for the Gmail account to a file
with open('credentials.pickle', 'wb') as token_file:
    pickle.dump(credentials, token_file)

# Load the credentials for the Gmail account from the file
with open('credentials.pickle', 'rb') as token_file:
    credentials = pickle.load(token_file)

# Create Gmail API service object
gmail_service = build('gmail', 'v1', credentials=credentials)

# Create Google Calendar API service object
calendar_service = build('calendar', 'v3', credentials=credentials)

# Function to fetch events from Google Calendar
def fetch_calendar_events(service):
    now = datetime.utcnow().isoformat() + 'Z'  # 'Z' indicates UTC time
    events_result = service.events().list(calendarId='primary', timeMin=now, maxResults=10, singleEvents=True, orderBy='startTime').execute()
    return events_result.get('items', [])

# Fetch events from the Google Calendar
events = fetch_calendar_events(calendar_service)

# Extract event information and format it for the GPT-3 prompt
event_strings = []
for event in events:
    start = event['start'].get('dateTime', event['start'].get('date'))
    summary = event['summary']
    event_string = f"{start}: {summary}"
    event_strings.append(event_string)

# Join event strings into a single string
events_input = "\n".join(event_strings)

priorities = """
1. Gym:
    - Goal: Improve physical fitness and mental wellbeing.
    - Task: Schedule a 90-minute gym session daily, ideally during the morning hours.
2. Assignments (Math-Econ major at UCSD):
    - Goal: Maintain a high GPA and complete assignments on time.
    - Task: Allocate dedicated time slots for each subject and assignment.
    - Spring Quarter 2023 Classes:
        - Math 18: Linear Algebra
        - Math 20D: Differential Equations
        - Econ 100C: Intermediate Microeconomics C
        - Chem 11: The Periodic Table
        - MGT 103: Product Marketing and Management
3. Business (DCUBE LLC):
    - Goal: Transition from an ecommerce company to an AI-driven company.
    - Task: Develop AI tools and services, including:
        - Data annotation
        - Web scraping
        - Economic and e-commerce forecasting
        - Data visualization
        - Data dashboards
        - Recommendation systems
    - Goal: Automate data collection and web scraping for MVP.
    - Task: Use GPT-4 to assist with the following activities:
        - Market research
        - SaaS toolkit development
        - Deployment and testing of tools
        - Website creation and marketing
        - Google Ads and other marketing strategies
        - Analyzing cost-to-profit ratios to stay competitive
    - Task: Complete the website and set up Upwork profiles for additional support.
    - Goal: Explore both service-based and software-as-a-service (SaaS) business models.
    - Task: Utilize GPT-4 to generate insights and strategies for scaling and growing the business
    - Task: Utilize GPT-4 to generate insights and strategies for the following activities:
        - Market research:
            - Analyzing current market trends and competition
            - Identifying main customer needs and pain points
        - SaaS toolkit development:
            - Suggesting features and functionalities
            - Generating a high-level architecture
        - Deployment and testing:
            - Best practices for deploying and maintaining a SaaS application
            - Setting up an automated testing framework
        - Website creation:
            - Designing a modern and user-friendly website layout
            - Creating engaging and SEO-optimized homepage copy
        - Marketing and advertising:
            - Generating ad copy ideas for a Google Ads campaign
            - Suggesting a content marketing strategy
        - Cost optimization and competitive analysis:
            - Analyzing the pricing strategy of top competitors
            - Suggesting a cost optimization strategy
        - Scaling and growing:
            - Identifying potential new markets and growth opportunities
            - Suggesting strategies for expanding into international markets
        - Additional purposes:
            - Creating engaging social media posts
            - Drafting an email campaign
            - Performing sentiment analysis on customer feedback
            - Designing a conversational flow for a chatbot
4. Gmail:
        Gmail:
        Goal: Organize inbox into categories and prioritize messages to improve productivity and response times.
        Context: I receive emails related to work, personal matters, newsletters, and promotions.
        Task: Create a structured and prioritized inbox that allows me to focus on important emails first, while keeping track of other messages.
        """

API_KEY = 'your openAI api key'


# Load your API key from an environment variable or secret management service
openai.api_key = API_KEY

messages = [
    {"role": "system", "content": "You are an AI Calendar/Secretary, AI Manager, and AI Consultant, responsible for managing the user's schedule, tasks, and providing guidance for their business."},
    {"role": "user", "content": "Please analyze my tasks, suggest specific time slots for each task today, provide recommendations for my business, and send an email with the suggestions and next steps. My tasks are:\n1. Gym (90-minute session)\n2. Assignments for Math-Econ major at UCSD\n3. Business tasks for my company, DCUBE LLC (include tasks such as using GPT-4 for market research, SaaS toolkit development, and deployment and testing of tools, completing the website, and setting up Upwork profiles)\n4. Organizing my Gmail inbox"}
]

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
    temperature=0.8,
    max_tokens=1500,
)

# Extract suggestions and next steps from the AI's response
suggestions_and_next_steps = response.choices[0].message['content'].strip()
print(suggestions_and_next_steps)

response_data = response
print(response_data)

assistant_message = response.choices[0].message
if assistant_message:
    print(assistant_message["content"].strip())
else:
    print(f"Error: {response_data['error']['message']}")

output = assistant_message["content"].strip()

# Function to parse the OpenAI output to get our Tasks and Times

def parse_gpt_output(output):
    task_lines = [line.strip() for line in output.split("\n") if line.strip().startswith("- ")]

    tasks = []
    for line in task_lines:
        time_range, summary = line.split(":", 1)
        time_range_parts = time_range.split("-")
        if len(time_range_parts) == 2:
            suggested_start_time = time_range_parts[0].strip()
            suggested_end_time = time_range_parts[1].strip()
            # Convert the suggested time to the appropriate format
            now = datetime.now()
            try:
                start_time_obj = datetime.strptime(suggested_start_time, '%I:%M %p')
                end_time_obj = datetime.strptime(suggested_end_time, '%I:%M %p')
            except ValueError:
                # If time format is not as expected, ignore the suggested time
                suggested_start_time = None
                suggested_end_time = None
            else:
                suggested_start_time = now.replace(hour=start_time_obj.hour, minute=start_time_obj.minute).strftime('%Y-%m-%dT%H:%M:%S')
                suggested_end_time = now.replace(hour=end_time_obj.hour, minute=end_time_obj.minute).strftime('%Y-%m-%dT%H:%M:%S')
        else:
            suggested_start_time = None
            suggested_end_time = None

        tasks.append({"summary": summary.strip(), "suggested_start_time": suggested_start_time, "suggested_end_time": suggested_end_time})

    return tasks


tasks = parse_gpt_output(suggestions_and_next_steps)
print(f"Extracted tasks: {tasks}")  # Print the extracted tasks

# Create a new event in Google Calendar for each suggested task

for task in tasks:
    print(f"Processing task: {task}")  # Print the task being processed
    task_summary, suggested_start_time, suggested_end_time = task["summary"], task["suggested_start_time"], task["suggested_end_time"]

    try:
        if suggested_start_time and suggested_end_time:
            start_time = datetime.strptime(suggested_start_time, "%Y-%m-%dT%H:%M:%S")
            end_time = datetime.strptime(suggested_end_time, "%Y-%m-%dT%H:%M:%S")

            event = {
                'summary': task_summary,
                'description': '',
                'start': {
                    'dateTime': start_time.isoformat(),
                    'timeZone': 'America/Los_Angeles',
                },
                'end': {
                    'dateTime': end_time.isoformat(),
                    'timeZone': 'America/Los_Angeles',
                },
                'reminders': {
                    'useDefault': True,
                },
            }

        else:
            date_today = datetime.now().date()
            event = {
                'summary': task_summary,
                'description': '',
                'start': {
                    'date': date_today.isoformat(),
                },
                'end': {
                    'date': (date_today + timedelta(days=1)).isoformat(),
                },
                'reminders': {
                    'useDefault': True,
                },
            }

        # Insert the event into the Google Calendar
        event = calendar_service.events().insert(calendarId='primary', body=event).execute()
        print(f"Event created: {event.get('htmlLink')}")
    except Exception as e:
        print(f"Unable to create calendar event for task: {task_summary}. Error: {str(e)}")  # Print the error if any

def send_email(subject, body):
    message = MIMEText(body)
    message['to'] = "youremail@host.suffix"
    message['subject'] = subject

    create_message = {'raw': base64.urlsafe_b64encode(message.as_bytes()).decode()}
    send_message = (gmail_service.users().messages().send(userId="me", body=create_message).execute())

    print(F'sent message to {message["to"]} Message Id: {send_message["id"]}')


# Send the suggestions and next steps as an email
send_email(subject="AI Calendar/Secretary, Manager, and Consultant Suggestions", body=suggestions_and_next_steps)

    
    # Function to fetch messages from Gmail
def fetch_gmail_messages(service):
    query = "is:unread"  # Modify the query as needed
    response = service.users().messages().list(userId='me', q=query).execute()
    messages = []
    if 'messages' in response:
        messages.extend(response['messages'])
    return messages

# Function to categorize messages
def categorize_messages(service, messages):
    categories = {
        "DCUBE": [],
        "School": [],
        "Personal": [],
        "Promotions": [],
    }

    for message in messages:
        msg = service.users().messages().get(userId='me', id=message['id']).execute()
        subject = ""
        from_email = ""
        for header in msg['payload']['headers']:
            if header['name'] == 'subject':
                subject = header['value']
            if header['name'] == 'From':
                from_email = header['value']

        # Modify the categorization logic as needed
        if "DCUBE" in subject or "DCUBE" in from_email:
            categories["DCUBE"].append(subject)
        elif "UCSD" in from_email or "TA" or "Professor" in subject:
            categories["School"].append(subject)
        elif "promo" in subject.lower() or "ebay" or "openboxecommerce" in from_email.lower():
            categories["Promotions"].append(subject)
        else:
            categories["Personal"].append(subject)

    return categories


def prioritize_messages(categories):
    prioritized_categories = []
    
    # Modify the order of categories as needed
    category_order = ["DCUBE", "School", "Personal", "Promotions"]
    
    for category in category_order:
        messages = categories[category]
        prioritized_messages = sorted(messages, key=lambda x: x.lower())
        prioritized_categories.extend(prioritized_messages)

    return prioritized_categories
# Fetch Gmail messages
messages = fetch_gmail_messages(gmail_service)

# Categorize Gmail messages
categorized_messages = categorize_messages(gmail_service, messages)

# Prioritize Gmail messages
prioritized_messages = prioritize_messages(categorized_messages)

