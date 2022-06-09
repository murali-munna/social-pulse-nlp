# Social Pulse - Understanding Brand Perception using Social Media Analytics
---
ECE229 (Spring 2022) Course Project by: Murali Dandu, Christopher Light, Hao Qiu, Liyang Ru, Pai Tong, Yen-Ju Tseng



## About
---
Listening to customers’ thoughts is one of the most important aspects for any brand growth. Social media analytics provides a powerful way to understand how they are perceiving your brand, industry and competitors. Here we are proposing ’Social Pulse’, a platform for Brand Marketers based on statistical text/NLP analytics.

This app analyzes the brand social media data in terms of sentiment, emotion, keywords and more!

## Installation
---



## User Story
---

A Brand Manager can ask and answer several questions like:
- How is my brand's sentiment over a specific time period?
- How is the user perception of the new features launched?
- What are the key-topics that users are talking about and how's their sentiment?
- How is the sentiment/perception varying across geographies and how can I allocate my budget and personnel accordingly?
- How are Twitter users different to Reddit and how can I perform targeting marketing based on that?
- How is my competitor brand performing?

## Data Extraction
---
**Twitter Data:** Scraped relevant hashtags using [Tweepy](https://www.tweepy.org/)

**Reddit Data:** Scraped relevant sub-reddits and search terms using [PRAW](https://praw.readthedocs.io/en/stable/)

## Models
---
* **Sentiment Detection:** 'Twitter and Reddit Sentimental analysis Dataset' modelled to detect Positive, Neutral and Negative classes. We used LinearSVC model with TF-IDF representations.

* **Emotion Detection:** 'Emotions Dataset' modelled to detect five emotions: joy, sadness, surprise, anger, fear. We used LinearSVC model with TF-IDF representations.

**Keyword Extraction:**

- **YAKE:** Unsupervised keyword extraction method based on text statistical features
- **KeyBERT:** Keyword extraction technique based on similarity between phrases (noun chunks) and document embeddings

## Dashboard Visualizations
---


## Application Architecture
---


## Documentation
---



