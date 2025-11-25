<script>
  import Youtube from "svelte-youtube-embed";
</script>

# judo footage analysis

This repository is work supporting _Semi-Supervised Extraction and Analysis of Judo Combat Phases from Recorded Live-Streamed Tournament Footage_.
The goal of the project is to automate live-stream recording segmentation into individual matches, extract combat phases from matches, and gather statistics at the tournament level.
See the [full YouTube playlist](https://youtube.com/playlist?list=PLaBtWXB-9VkbHSHyyY-fjAVD7dO1P2PdO&si=muL92HcCVlvfbQH9) for demonstration of the project.


This project was part of CS8813 Introduction to Research at Georgia Tech Europe during the Spring 2024 semester.

## Annotating Judo Matches

One of the more challenging aspects of the project was annotating the judo matches.
There is a limited amount of labeled data available for the task that we wanted to achieve, so we had to go through the process of building our own dataset.

<Youtube id="50v9sShpuUw" />

## Finding out when a match starts

One of the first steps in the project was to find out when a match starts.
We dumped out images of videos across all of the mats and built a model to sort each frame into different categories.

<Youtube id="aVgAX6BmLCg" />
