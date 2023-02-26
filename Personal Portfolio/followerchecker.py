#!/usr/bin/env python
# coding: utf-8

# In[11]:


# Affaan's Instagram Follower Checker Tool to see who you are 
# following that isn't following you back.


# First login to instagram through creating an instance in your terminal
# instaloader --l YOUR-USERNAME
# Then download this script as a .py file and call it in your terminal using python3 followerchecker.py

import instaloader
import pandas as pd

# Get instance
L = instaloader.Instaloader()

# Account Details
USER = 'username'
PASSWORD = 'password'

# Now that the instance is created Login again
L.login(USER, PASSWORD)        # (login)
L.interactive_login(USER)      # (ask password on terminal)
L.load_session_from_file(USER) # (load session created w/
                               #  `instaloader -l USERNAME`)

profile = instaloader.Profile.from_username(L.context, USER)

# creates following and followers and then creates 2 sets out of them

following = set([x.username for x in profile.get_followees()])
followers = set([x.username for x in profile.get_followers()])

# compares these dataframes and gets just the people not following you back

not_following_back = following - followers

with open('not_following_back.txt', 'w') as file:
    file.write('People you are following who are not following you back:\n')
    for user in not_following_back:
        file.write(user + '\n')

L.context.log("Done.")

# once this is complete check your user files /users/'your computer username' the file should be here
# a text file named not_following_back and in this text file has the usernames of all the people
# who do not follow you back
# hope you guys like this program be sure to follow my socials and github
# github: affaan-m
# instagram: affaan_rm
# linkedin: https://www.linkedin.com/in/affaanmustafa/
# https://linktr.ee/affaan.eth
# Check out my Company DCUBE @ dcube.ai : Personalized and Affordable Data Annotation Services

