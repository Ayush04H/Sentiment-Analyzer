# Sentiment Analysis Dashboard

The **Sentiment Analysis Dashboard** is a web-based application designed to analyze the sentiment of text data using machine learning techniques. The application provides an API for analyzing text sentiment, generating visual graphs based on sentiment scores, and presenting a dashboard for aggregated sentiment analysis. The backend is built with FastAPI and MongoDB for data storage.

## API Endpoints

The following API endpoints are available for interacting with the Sentiment Analysis Dashboard:

- **POST /analyze/**: Accepts text input and returns sentiment analysis scores (positive, neutral, negative) using a machine learning model. The sentiment analysis result is stored in a MongoDB database.

- **GET /graph/**: Accepts a `text_id` and generates a sentiment graph based on the sentiment scores of the corresponding text. The graph is returned as an image in the response.

- **GET /dashboard/**: Aggregates all sentiment scores stored in the database and generates a visual dashboard of the overall sentiment distribution.

- **GET /**: Provides a welcome message with instructions on how to use the various API endpoints.

## MongoDB Export and Import Instructions using MongoDB Compass

This guide provides instructions on how to export a MongoDB database to a local folder using MongoDB Compass, push it to GitHub, and import it on another machine.

# MongoDB Export and Import Instructions using MongoDB Compass

This guide provides instructions on how to export a MongoDB database to a local folder using MongoDB Compass, push it to GitHub, and import it on another machine.

## 1. Exporting the MongoDB Database Using MongoDB Compass

### Step 1: Open MongoDB Compass
- Launch MongoDB Compass and connect to your local or remote MongoDB server.

### Step 2: Select the Database
- After connecting, navigate to the database you want to export (e.g., `sentiment_db`).
- Click on the database name in the sidebar to view the collections inside.

### Step 3: Export the Collection
- Choose a collection you wish to export (e.g., `texts` or `images`).
- Click on the collection to open its documents.
- At the top-right corner, click on the **Export Collection** button.

### Step 4: Choose Export Format
- Select either `JSON` or `CSV` format for the export. For databases with binary data (e.g., images), `JSON` is the recommended format.
- Choose whether to export all documents or apply filters to export specific data.

### Step 5: Select Destination Folder
- Choose the folder on your local machine where you want to save the exported file(s). Each collection will be saved as a separate file.

### Step 6: Export the Collection
- Once the format and destination are selected, click **Export**. The file will be saved to the specified location.
- Repeat this process for all the collections in the database.

## 2. Pushing the Exported Data to GitHub

### Step 1: Initialize a Git Repository
- Navigate to the local folder where the exported files are saved.
- Open a terminal and run the following commands to initialize a Git repository and push the files to GitHub:

```bash
git init
git add .
git commit -m "Initial commit with MongoDB export"
git remote add origin https://github.com/yourusername/your-repo.git
git push -u origin main


```
## 3. Importing from Databse Folder

### Step 1: Download the Exported Files

### Step 2: Open MongoDB Compass

### Step 3: Select the Database
- In MongoDB Compass, create a new database if it does not already exist. You can name it the same as the original (e.g., sentiment_db).

### Step 4: Import the Collection
- Click on the desired database and select Create Collection to create the same collections (e.g., texts, images) as in the original database.
- After creating a collection, click on it to open its documents view
- At the Add-File dropdown button corner, click on Import Data.
- Select the corresponding exported file (JSON or CSV) from your local machine.
- Set the correct file format (JSON or CSV) in the import options.
- Click Import. Repeat this step for each collection.



# To Run Main File:

```bash

cd backend
uvicorn run main:app reload

```

## Ensure that the MongoDB Server is UP


