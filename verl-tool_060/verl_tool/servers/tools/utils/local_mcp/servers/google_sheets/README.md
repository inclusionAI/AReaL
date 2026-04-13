# mcp-google-sheets: A Google Sheets MCP server

This MCP server integrates with your Google Drive and Google Sheets, to enable creating and modifying spreadsheets.

## Overview

A Model Context Protocol server for interacting with Google Sheets. This server provides tools to create, read, update, and manage spreadsheets through the Google Sheets API.

### Tools

1. `get_sheet_data`
   - Get data from a specific sheet in a Google Spreadsheet
   - Input:
     - `spreadsheet_id` (string): The ID of the spreadsheet (found in the URL)
     - `sheet` (string): The name of the sheet
     - `range` (optional string): Cell range in A1 notation (e.g., 'A1:C10')
   - Returns: A 2D array of the sheet data

2. `update_cells`
   - Update cells in a Google Spreadsheet
   - Input:
     - `spreadsheet_id` (string): The ID of the spreadsheet
     - `sheet` (string): The name of the sheet
     - `range` (string): Cell range in A1 notation
     - `data` (2D array): Values to update
   - Returns: Result of the update operation

3. `batch_update_cells`
   - Batch update multiple ranges in a Google Spreadsheet
   - Input:
     - `spreadsheet_id` (string): The ID of the spreadsheet
     - `sheet` (string): The name of the sheet
     - `ranges` (object): Dictionary mapping range strings to 2D arrays of values
   - Returns: Result of the batch update operation

4. `list_sheets`
   - List all sheets in a Google Spreadsheet
   - Input:
     - `spreadsheet_id` (string): The ID of the spreadsheet
   - Returns: List of sheet names

5. `list_spreadsheets`
   - List all spreadsheets in the configured Google Drive folder
   - Returns: List of spreadsheets with their ID and title
   - Note: If using service account authentication, this will list spreadsheets in the shared folder

6. `create_spreadsheet`
   - Create a new Google Spreadsheet
   - Input:
     - `title` (string): The title of the new spreadsheet
   - Returns: Information about the newly created spreadsheet including its ID
   - Note: When using service account authentication with a configured folder ID, the spreadsheet will be created in that folder

7. `create_sheet`
   - Create a new sheet tab in an existing Google Spreadsheet
   - Input:
     - `spreadsheet_id` (string): The ID of the spreadsheet
     - `title` (string): The title for the new sheet
   - Returns: Information about the newly created sheet

8. `get_spreadsheet_info`
   - Get basic information about a Google Spreadsheet.
   - Input:
     - `spreadsheet_id` (string): The ID of the spreadsheet
   - Returns: JSON string with spreadsheet information

9. Additional tools: `add_rows`, `add_columns`, `copy_sheet`, `rename_sheet`

## Installation and Setup

The server requires some setup in Google Cloud Platform and choosing an authentication method before running.

### Google Cloud Platform Setup (Required for All Methods)

1. Create a Google Cloud Platform project:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select an existing one
   - Enable the Google Sheets API and Google Drive API

### Choose an Authentication Method

You can use one of two authentication methods:

#### Method 1: Service Account Authentication (Recommended, Non-interactive)

Service accounts provide headless authentication without browser prompts, ideal for automated or server environments. Benefits include:
- No browser interaction needed for authentication
- Works well in headless environments
- Authentication doesn't expire as frequently as OAuth tokens
- Perfect for server deployments and automation

Setup steps:

1. Create a service account:
   - Go to Google Cloud Console → IAM & Admin → Service Accounts
   - Create a new service account with a descriptive name
   - Grant it appropriate roles (for Google Sheets access)
   - Create and download a JSON key file

2. Create a dedicated folder in Google Drive to share with the service account:
   - Go to Google Drive and create a new folder (e.g., "Claude Sheets")
   - Note the folder's ID from its URL: `https://drive.google.com/drive/folders/FOLDER_ID_HERE`
   - Right-click the folder and select "Share"
   - Share it with the service account email address (found in the JSON file as `client_email`)
   - Give it "Editor" access

3. Set these environment variables:
   - `SERVICE_ACCOUNT_PATH`: Path to service account JSON key file
   - `DRIVE_FOLDER_ID`: ID of the Google Drive folder shared with the service account

#### Method 2: OAuth 2.0 Authentication (Interactive)

This method requires browser interaction for the first-time setup, suitable for personal use or development.

1. Configure OAuth for your project:
   - Configure the OAuth consent screen
   - Create OAuth 2.0 Client ID credentials (Desktop application type)
   - Download the credentials JSON file and save it as `credentials.json`

2. Set these environment variables:
   - `CREDENTIALS_PATH`: Path to the downloaded OAuth credentials file (default: `credentials.json`)
   - `TOKEN_PATH`: Path where the authentication token will be stored (default: `token.json`)

### Setting Environment Variables

```bash
# For service account authentication (recommended)
export SERVICE_ACCOUNT_PATH=/path/to/your/service-account-key.json
export DRIVE_FOLDER_ID=your_shared_folder_id_here

# OR for OAuth authentication
export CREDENTIALS_PATH=/path/to/your/credentials.json
export TOKEN_PATH=/path/to/your/token.json
```