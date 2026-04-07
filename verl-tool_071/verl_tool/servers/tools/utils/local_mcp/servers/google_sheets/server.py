"""
Google Spreadsheet MCP Server.
Inspired from:
1. https://github.com/kazz187/mcp-google-spreadsheet/tree/main
2. https://github.com/xing5/mcp-google-sheets/tree/main
"""
# pylint: disable=broad-exception-caught,unused-argument,redefined-builtin
import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

import click
from mcp.server.fastmcp import FastMCP, Context
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from mcpuniverse.common.logger import get_logger

# Constants
SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
TOKEN_PATH = os.environ.get('TOKEN_PATH', 'token.json')
CREDENTIALS_PATH = os.environ.get('CREDENTIALS_PATH', 'credentials.json')
SERVICE_ACCOUNT_PATH = os.environ.get('SERVICE_ACCOUNT_PATH', 'service_account.json')
DRIVE_FOLDER_ID = os.environ.get('DRIVE_FOLDER_ID', '')  # Working directory in Google Drive


@dataclass
class SpreadsheetContext:
    """Context for Google Spreadsheet service"""
    sheets_service: Any
    drive_service: Any
    folder_id: Optional[str] = None


@asynccontextmanager
async def spreadsheet_lifespan(server: FastMCP) -> AsyncIterator[SpreadsheetContext]:
    """Manage Google Spreadsheet API connection lifecycle"""
    # Authenticate and build the service
    creds = None
    # Check for service account authentication first
    if SERVICE_ACCOUNT_PATH and os.path.exists(SERVICE_ACCOUNT_PATH):
        try:
            # Regular service account authentication
            creds = service_account.Credentials.from_service_account_file(
                SERVICE_ACCOUNT_PATH,
                scopes=SCOPES
            )
            print("Using service account authentication")
            print(f"Working with Google Drive folder ID: {DRIVE_FOLDER_ID or 'Not specified'}")
        except Exception as e:
            print(f"Error using service account authentication: {e}")
            print("Falling back to OAuth flow")
            creds = None

    # Fall back to OAuth flow if service account auth failed or not configured
    if not creds:
        print("Using OAuth authentication flow")
        if os.path.exists(TOKEN_PATH):
            with open(TOKEN_PATH, "r", encoding="utf-8") as token:
                creds = Credentials.from_authorized_user_info(json.load(token), SCOPES)

        # If credentials are not valid or don't exist, get new ones
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
                creds = flow.run_local_server(port=0)

            # Save the credentials for the next run
            with open(TOKEN_PATH, "w", encoding="utf-8") as token:
                token.write(creds.to_json())

    # Build the services
    sheets_service = build('sheets', 'v4', credentials=creds)
    drive_service = build('drive', 'v3', credentials=creds)
    try:
        # Provide the service in the context
        yield SpreadsheetContext(
            sheets_service=sheets_service,
            drive_service=drive_service,
            folder_id=DRIVE_FOLDER_ID if DRIVE_FOLDER_ID else None
        )
    finally:
        # No explicit cleanup needed for Google APIs
        pass


# Initialize the MCP server with lifespan management
mcp = FastMCP("Google Spreadsheet",
              dependencies=["google-auth", "google-auth-oauthlib", "google-api-python-client"],
              lifespan=spreadsheet_lifespan)


@mcp.tool()
async def get_sheet_data(
        spreadsheet_id: str,
        sheet: str,
        range: Optional[str] = None,
        ctx: Context = None
) -> List[List[Any]]:
    """
    Get data from a specific sheet in a Google Spreadsheet.

    Args:
        spreadsheet_id: The ID of the spreadsheet (found in the URL)
        sheet: The name of the sheet
        range: Optional cell range in A1 notation (e.g., 'A1:C10'). If not provided, gets all data.

    Returns:
        A 2D array of the sheet data
    """
    sheets_service = ctx.request_context.lifespan_context.sheets_service

    # Construct the range
    if range:
        full_range = f"{sheet}!{range}"
    else:
        full_range = sheet

    # Call the Sheets API
    result = sheets_service.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id,
        range=full_range
    ).execute()

    # Get the values from the response
    values = result.get('values', [])
    return values


@mcp.tool()
async def update_cells(
        spreadsheet_id: str,
        sheet: str,
        range: str,
        data: List[List[Any]],
        ctx: Context = None
) -> Dict[str, Any]:
    """
    Update cells in a Google Spreadsheet.

    Args:
        spreadsheet_id: The ID of the spreadsheet (found in the URL)
        sheet: The name of the sheet
        range: Cell range in A1 notation (e.g., 'A1:C10')
        data: 2D array of values to update

    Returns:
        Result of the update operation
    """
    sheets_service = ctx.request_context.lifespan_context.sheets_service

    # Construct the range
    full_range = f"{sheet}!{range}"

    # Prepare the value range object
    value_range_body = {
        'values': data
    }

    # Call the Sheets API to update values
    return sheets_service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range=full_range,
        valueInputOption='USER_ENTERED',
        body=value_range_body
    ).execute()


@mcp.tool()
async def batch_update_cells(
        spreadsheet_id: str,
        sheet: str,
        ranges: Dict[str, List[List[Any]]],
        ctx: Context = None
) -> Dict[str, Any]:
    """
    Batch update multiple ranges in a Google Spreadsheet.

    Args:
        spreadsheet_id: The ID of the spreadsheet (found in the URL)
        sheet: The name of the sheet
        ranges: Dictionary mapping range strings to 2D arrays of values
               e.g., {'A1:B2': [[1, 2], [3, 4]], 'D1:E2': [['a', 'b'], ['c', 'd']]}

    Returns:
        Result of the batch update operation
    """
    sheets_service = ctx.request_context.lifespan_context.sheets_service

    # Prepare the batch update request
    data = []
    for range_str, values in ranges.items():
        full_range = f"{sheet}!{range_str}"
        data.append({
            'range': full_range,
            'values': values
        })

    batch_body = {
        'valueInputOption': 'USER_ENTERED',
        'data': data
    }
    return sheets_service.spreadsheets().values().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body=batch_body
    ).execute()


@mcp.tool()
async def add_rows(
        spreadsheet_id: str,
        sheet: str,
        count: int,
        start_row: Optional[int] = None,
        ctx: Context = None
) -> Dict[str, Any]:
    """
    Add rows to a sheet in a Google Spreadsheet.

    Args:
        spreadsheet_id: The ID of the spreadsheet (found in the URL)
        sheet: The name of the sheet
        count: Number of rows to add
        start_row: 0-based row index to start adding. If not provided, adds at the end.

    Returns:
        Result of the operation
    """
    sheets_service = ctx.request_context.lifespan_context.sheets_service

    # Get sheet ID
    spreadsheet = sheets_service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    sheet_id = None

    for s in spreadsheet['sheets']:
        if s['properties']['title'] == sheet:
            sheet_id = s['properties']['sheetId']
            break

    if sheet_id is None:
        return {"error": f"Sheet '{sheet}' not found"}

    # Prepare the insert rows request
    request_body = {
        "requests": [
            {
                "insertDimension": {
                    "range": {
                        "sheetId": sheet_id,
                        "dimension": "ROWS",
                        "startIndex": start_row if start_row is not None else 0,
                        "endIndex": (start_row if start_row is not None else 0) + count
                    },
                    "inheritFromBefore": start_row is not None and start_row > 0
                }
            }
        ]
    }
    return sheets_service.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body=request_body
    ).execute()


@mcp.tool()
async def add_columns(
        spreadsheet_id: str,
        sheet: str,
        count: int,
        start_column: Optional[int] = None,
        ctx: Context = None
) -> Dict[str, Any]:
    """
    Add columns to a sheet in a Google Spreadsheet.

    Args:
        spreadsheet_id: The ID of the spreadsheet (found in the URL)
        sheet: The name of the sheet
        count: Number of columns to add
        start_column: 0-based column index to start adding. If not provided, adds at the end.

    Returns:
        Result of the operation
    """
    sheets_service = ctx.request_context.lifespan_context.sheets_service

    # Get sheet ID
    spreadsheet = sheets_service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    sheet_id = None

    for s in spreadsheet['sheets']:
        if s['properties']['title'] == sheet:
            sheet_id = s['properties']['sheetId']
            break

    if sheet_id is None:
        return {"error": f"Sheet '{sheet}' not found"}

    # Prepare the insert columns request
    request_body = {
        "requests": [
            {
                "insertDimension": {
                    "range": {
                        "sheetId": sheet_id,
                        "dimension": "COLUMNS",
                        "startIndex": start_column if start_column is not None else 0,
                        "endIndex": (start_column if start_column is not None else 0) + count
                    },
                    "inheritFromBefore": start_column is not None and start_column > 0
                }
            }
        ]
    }
    return sheets_service.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body=request_body
    ).execute()


@mcp.tool()
async def list_sheets(spreadsheet_id: str, ctx: Context = None) -> List[str]:
    """
    List all sheets in a Google Spreadsheet.

    Args:
        spreadsheet_id: The ID of the spreadsheet (found in the URL)

    Returns:
        List of sheet names
    """
    sheets_service = ctx.request_context.lifespan_context.sheets_service

    # Get spreadsheet metadata
    spreadsheet = sheets_service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()

    # Extract sheet names
    sheet_names = [sheet['properties']['title'] for sheet in spreadsheet['sheets']]
    return sheet_names


@mcp.tool()
async def copy_sheet(
        src_spreadsheet: str,
        src_sheet: str,
        dst_spreadsheet: str,
        dst_sheet: str,
        ctx: Context = None
) -> Dict[str, Any]:
    """
    Copy a sheet from one spreadsheet to another.

    Args:
        src_spreadsheet: Source spreadsheet ID
        src_sheet: Source sheet name
        dst_spreadsheet: Destination spreadsheet ID
        dst_sheet: Destination sheet name

    Returns:
        Result of the operation
    """
    sheets_service = ctx.request_context.lifespan_context.sheets_service

    # Get source sheet ID
    src = sheets_service.spreadsheets().get(spreadsheetId=src_spreadsheet).execute()
    src_sheet_id = None

    for s in src['sheets']:
        if s['properties']['title'] == src_sheet:
            src_sheet_id = s['properties']['sheetId']
            break

    if src_sheet_id is None:
        return {"error": f"Source sheet '{src_sheet}' not found"}

    # Copy the sheet to destination spreadsheet
    copy_result = sheets_service.spreadsheets().sheets().copyTo(
        spreadsheetId=src_spreadsheet,
        sheetId=src_sheet_id,
        body={
            "destinationSpreadsheetId": dst_spreadsheet
        }
    ).execute()

    # If destination sheet name is different from the default copied name, rename it
    if 'title' in copy_result and copy_result['title'] != dst_sheet:
        # Get the ID of the newly copied sheet
        copy_sheet_id = copy_result['sheetId']

        # Rename the copied sheet
        rename_request = {
            "requests": [
                {
                    "updateSheetProperties": {
                        "properties": {
                            "sheetId": copy_sheet_id,
                            "title": dst_sheet
                        },
                        "fields": "title"
                    }
                }
            ]
        }
        rename_result = sheets_service.spreadsheets().batchUpdate(
            spreadsheetId=dst_spreadsheet,
            body=rename_request
        ).execute()
        return {"copy": copy_result, "rename": rename_result}

    return {"copy": copy_result}


@mcp.tool()
async def rename_sheet(
        spreadsheet: str,
        sheet: str,
        new_name: str,
        ctx: Context = None
) -> Dict[str, Any]:
    """
    Rename a sheet in a Google Spreadsheet.

    Args:
        spreadsheet: Spreadsheet ID
        sheet: Current sheet name
        new_name: New sheet name

    Returns:
        Result of the operation
    """
    sheets_service = ctx.request_context.lifespan_context.sheets_service

    # Get sheet ID
    spreadsheet_data = sheets_service.spreadsheets().get(spreadsheetId=spreadsheet).execute()
    sheet_id = None

    for s in spreadsheet_data['sheets']:
        if s['properties']['title'] == sheet:
            sheet_id = s['properties']['sheetId']
            break

    if sheet_id is None:
        return {"error": f"Sheet '{sheet}' not found"}

    # Prepare the rename request
    request_body = {
        "requests": [
            {
                "updateSheetProperties": {
                    "properties": {
                        "sheetId": sheet_id,
                        "title": new_name
                    },
                    "fields": "title"
                }
            }
        ]
    }
    return sheets_service.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet,
        body=request_body
    ).execute()


@mcp.tool()
async def get_spreadsheet_info(spreadsheet_id: str) -> str:
    """
    Get basic information about a Google Spreadsheet.

    Args:
        spreadsheet_id: The ID of the spreadsheet

    Returns:
        JSON string with spreadsheet information
    """
    # Access the context through mcp.get_lifespan_context() for resources
    context = mcp.get_lifespan_context()
    sheets_service = context.sheets_service

    # Get spreadsheet metadata
    spreadsheet = sheets_service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()

    # Extract relevant information
    info = {
        "title": spreadsheet.get('properties', {}).get('title', 'Unknown'),
        "sheets": [
            {
                "title": sheet['properties']['title'],
                "sheetId": sheet['properties']['sheetId'],
                "gridProperties": sheet['properties'].get('gridProperties', {})
            }
            for sheet in spreadsheet.get('sheets', [])
        ]
    }
    return json.dumps(info, indent=2)


@mcp.tool()
async def create_spreadsheet(title: str, ctx: Context = None) -> Dict[str, Any]:
    """
    Create a new Google Spreadsheet.

    Args:
        title: The title of the new spreadsheet

    Returns:
        Information about the newly created spreadsheet including its ID
    """
    sheets_service = ctx.request_context.lifespan_context.sheets_service
    drive_service = ctx.request_context.lifespan_context.drive_service
    folder_id = ctx.request_context.lifespan_context.folder_id

    # Create the spreadsheet using Sheets API
    spreadsheet_body = {
        'properties': {
            'title': title
        }
    }

    # Create the spreadsheet
    spreadsheet = sheets_service.spreadsheets().create(
        body=spreadsheet_body,
        fields='spreadsheetId,properties,sheets'
    ).execute()

    spreadsheet_id = spreadsheet.get('spreadsheetId')
    print(f"Spreadsheet created with ID: {spreadsheet_id}")

    # If a folder_id is specified, move the spreadsheet to that folder
    if folder_id:
        try:
            # Get the current parents
            file = drive_service.files().get(
                fileId=spreadsheet_id,
                fields='parents'
            ).execute()

            previous_parents = ",".join(file.get('parents', []))

            # Move the file to the specified folder
            drive_service.files().update(
                fileId=spreadsheet_id,
                addParents=folder_id,
                removeParents=previous_parents,
                fields='id, parents'
            ).execute()

            print(f"Spreadsheet moved to folder with ID: {folder_id}")
        except Exception as e:
            print(f"Warning: Could not move spreadsheet to folder: {e}")

    return {
        'spreadsheetId': spreadsheet_id,
        'title': spreadsheet.get('properties', {}).get('title', title),
        'sheets': [sheet.get('properties', {}).get('title', 'Sheet1') for sheet in spreadsheet.get('sheets', [])],
        'folder': folder_id if folder_id else 'root'
    }


@mcp.tool()
async def create_sheet(
        spreadsheet_id: str,
        title: str,
        ctx: Context = None
) -> Dict[str, Any]:
    """
    Create a new sheet tab in an existing Google Spreadsheet.

    Args:
        spreadsheet_id: The ID of the spreadsheet
        title: The title for the new sheet

    Returns:
        Information about the newly created sheet
    """
    sheets_service = ctx.request_context.lifespan_context.sheets_service

    # Define the add sheet request
    request_body = {
        "requests": [
            {
                "addSheet": {
                    "properties": {
                        "title": title
                    }
                }
            }
        ]
    }

    # Execute the request
    result = sheets_service.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body=request_body
    ).execute()

    # Extract the new sheet information
    new_sheet_props = result['replies'][0]['addSheet']['properties']
    return {
        'sheetId': new_sheet_props['sheetId'],
        'title': new_sheet_props['title'],
        'index': new_sheet_props.get('index'),
        'spreadsheetId': spreadsheet_id
    }


@mcp.tool()
async def list_spreadsheets(ctx: Context = None) -> List[Dict[str, str]]:
    """
    List all spreadsheets in the configured Google Drive folder.
    If no folder is configured, lists spreadsheets from 'My Drive'.

    Returns:
        List of spreadsheets with their ID and title
    """
    drive_service = ctx.request_context.lifespan_context.drive_service
    folder_id = ctx.request_context.lifespan_context.folder_id

    query = "mimeType='application/vnd.google-apps.spreadsheet'"

    # If a specific folder is configured, search only in that folder
    if folder_id:
        query += f" and '{folder_id}' in parents"
        print(f"Searching for spreadsheets in folder: {folder_id}")
    else:
        print("Searching for spreadsheets in 'My Drive'")

    # List spreadsheets
    results = drive_service.files().list(
        q=query,
        spaces='drive',
        fields='files(id, name)',
        orderBy='modifiedTime desc'
    ).execute()

    spreadsheets = results.get('files', [])
    return [{'id': sheet['id'], 'title': sheet['name']} for sheet in spreadsheets]


@click.command()
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    default="stdio",
    help="Transport type",
)
@click.option("--port", default="8000", help="Port to listen on for SSE")
def main(transport: str, port: str):
    """
    Starts the initialized MCP server.

    :param port: Port for SSE.
    :param transport: The transport type, e.g., `stdio` or `sse`.
    :return:
    """
    assert transport.lower() in ["stdio", "sse"], \
        "Transport should be `stdio` or `sse`"
    logger = get_logger("Service:google-sheets")
    logger.info("Starting the MCP server")
    mcp.run(transport=transport.lower())
