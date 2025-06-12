import os
import json
import logging
from typing import List, Dict, TypedDict, Literal

import google.generativeai as genai
from langgraph.graph import StateGraph, END

from tools import WebSearchTool, SEDSearchTool
from sed_agent import SEDAgent

# Get logger instance
logger = logging.getLogger(__name__)

# --- State Definition ---
class MainAgentState(TypedDict):
    """Defines the state for the main orchestrator agent."""
    messages: List[Dict[str, str]]
    tool_choice: Literal["web_search", "sed_search", "none"]
    search_query: str
    search_results: List[Dict[str, str]]
    sed_summary: str

"""
curl -X GET -H "X-API-Key: eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJpYXQiOjE3NDk3NTI0NzIsImF1ZCI6ImFwaTovL2lnbml0ZSIsImlzcyI6IjAwdWI2eXlydWNmZ3lXWnRCNWQ3OjBvYTlmdHBnc3dTZTBlMDNtNWQ3OnRlc3RlciIsImV4cCI6MzMyODU3NTI0NzIsInZlciI6MSwianRpIjoiQVQuWHdoOG1yNkxTaFJ6YU95eFVYQzlrdDFBZnFEMGQ0T2lmckt0Y2RsZzA3VS5vYXIzZmFxaHN5NG9wUE5iZzVkNyIsImNpZCI6IjBvYTlmdHBnc3dTZTBlMDNtNWQ3IiwidWlkIjoiMDB1YjZ5eXJ1Y2ZneVdadEI1ZDciLCJzY3AiOlsib2ZmbGluZV9hY2Nlc3MiLCJwcm9maWxlIiwiZW1haWwiLCJvcGVuaWQiXSwiYXV0aF90aW1lIjoxNzQ5NzMzODAwLCJzdWIiOiJkc3Ryb3VkQGZsYXNocG9pbnQtaW50ZWwuY29tIiwidXNuIjoiZHN0cm91ZEBmbGFzaHBvaW50LWludGVsLmNvbSIsImZmcyI6W10sInNmX2lkIjoiMDAxbzAwMDAwMHlQWFh1QUFPIiwib3JnX2lkIjo1NSwidXNyX2lkIjoiMDB1YjZ5eXJ1Y2ZneVdadEI1ZDciLCJvZ24iOiJGbGFzaHBvaW50IiwiZ3JvdXBzIjpbIklHTklURS5DQ01DX0hPU1RfQVRUUklCVVRFUy5VU0VSIiwiRUNIT1NFQy5OU0kuVVNFUiIsIklHTklURS5DQ01DX0FQSV9BTkRfVUkuVVNFUiIsIklHTklURS5CUkFORF9JTlRFTC5JTlRFUk5BTCIsIklHTklURS5OU0lfU0VELklOVEVSTkFMIiwiRmxhc2hwb2ludCIsIklHTklURS5CUkFORF9JTlRFTF9QT1YuSU5URVJOQUwiLCJJR05JVEUuU1VQRVIuSU5URVJOQUwiLCJFQ0hPU0VDLlNVUEVSLklOVEVSTkFMIiwiRUNIT1NFQy5IT1NUQkFTRUQuVVNFUiIsIlZVTE5EQl9PUkdfVVNFUiIsIkVDSE9TRUMuQkVBQ09OLlVTRVIiLCJJR05JVEUuRlJBVURfSU5URUwuVVNFUiIsIkV2ZXJ5b25lIiwiRUNIT1NFQy5CQVNJQy5VU0VSIiwiTUZBIE9wdGVkLWluIiwiSUdOSVRFLlZVTE5fQVBJLlVTRVIiLCJJR05JVEUuQ0NNQ19IT1NUX0RBVEEuVVNFUiIsIkVDSE9TRUMuVFdJVFRFUi5VU0VSIiwiUmVhZE1lIEFkbWlucyIsIlRPU19TSUdORUQiLCJJREVOVElGWS5CQVNJQy5VU0VSIiwiSUdOSVRFLkNDTV9IT1NUQkFTRURfQVBJLlVTRVIiLCJFQ0hPU0VDLkFQQUNfREFUQS5VU0VSIiwiSUdOSVRFLkJSQU5EX0lOVEVMLlVTRVIiLCJJR05JVEUuVlVMTl9QUkVNSVVNLlVTRVIiLCJNQU5BR0VEX0FUVFJJQlVUSU9OLkRFVkVMT1BNRU5ULklOVEVSTkFMIiwiRUNIT1NFQy5NRUFfREFUQS5VU0VSIiwiSUdOSVRFLk5TSV9TRUQuVVNFUiIsIklHTklURS5CQVNJQy5VU0VSIiwiTUFOQUdFRF9BVFRSSUJVVElPTi5ERU1PLlRSSUFMIl0sInBybSI6WyJJR05JVEVfVUkiLCJJR05JVEVfQVBJIiwiQ1RJX1JGSV9DUlVEIiwiQ1RJX1JGSV9BRE1JTiIsIkNUSV9HRU5fQ09MTEVDVCIsIkNUSV9SQU5TT01XQVJFIiwiVlVMTl9QUk9EVUNUU19PUkctVVNFUiIsIlZVTE5fVkVORE9SU19PUkctVVNFUiIsIlZVTE5fVlVMTkVSQUJJTElUSUVTX09SRy1VU0VSIiwiVlVMTl9NRU5USU9OU19PUkctVVNFUiIsIlZVTE5fQUxFUlQtUlVMRVNfT1JHLVVTRVIiLCJWVUxOX0FMRVJUU19PUkctVVNFUiIsImRhdC5lZG0uaGQub3JnLnIiLCJkYXQuZWRtLm9yZy5yIiwiZGF0LmluZC5yIiwiZGF0Lm1lZC5kIiwiZGF0Lm1rdC5yIiwiVlVMTl9DVElfUFJFTUlVTSIsIlZVTE5fQ1RJX0JBU0lDIiwiVlVMTl9BTEVSVC1SVUxFU19TVVBFUi1BRE1JTiIsIlZVTE5fQUxFUlRTX1NVUEVSLUFETUlOIiwiVlVMTl9BTEVSVC1URU1QTEFURVNfU1VQRVItQURNSU4iLCJJR05JVEVfQ1RJX1JFUE9SVFMiLCJJR05JVEVfUFNJX1JFUE9SVFMiLCJJR05JVEVfVlVMTl9SRVBPUlRTIiwiSUdOSVRFX0ZSQVVEX1JFUE9SVFMiLCJJR05JVEVfTkFUU0VDX1JFUE9SVFMiLCJkYXQuZWRtLmhkLnIiLCJkYXQuZWRtLm9yZy53IiwiZGF0Lmdscy5yIiwiZGF0LmVkbS5yIiwiZGF0LnJlcC5hc3MuciIsIklHTklURV9BTEVSVElOR19BRE1JTiIsImRhdC51c3IuciIsImRhdC51c3IudyIsIkNUSV9BU1NFVF9DUlVEIiwiQ1RJX0FTU0VUX1IiLCJkYXQuZG0uciIsImRhdC5kbS53IiwiZGF0LmRtLmFkbS5yIiwiZGF0LmRtLmFkbS53IiwiZGF0LmRtLnBvdi53IiwiZGF0LnJlcC53IiwiZGF0LnJlcC5yIiwiZGF0LnJlcC5hc3MudyIsImRhdC50b3AudyIsImRhdC5lZG0udyIsIkZJUkVIT1NFX0RFRkFVTFRfREFUQSIsIkZJUkVIT1NFX1NFTlNJVElWRV9EQVRBIiwiZGF0LmNjbS5jdXMuciIsImRhdC5jY20uY3VzLmhkLnIiLCJkYXQuZGVhLnIiLCJkYXQuZXAuY3JlZC5yIiwiZGF0LmVwLmNyZWQudyIsImRhdC5jc2IuciIsImRhdC5jZm0uYmEuciIsImRhdC5jZm0uciIsImRhdC5jZm0udyIsIkNDTUMuVUkiLCJkYXQuY2NtLmN1cy5oYS5yIiwiZGF0LmNjbS5jdXMuaGEuci5iZXRhIiwiZGF0LmNjbS5jdXMuci5iZXRhIiwiZGF0LmNjbS5jdXMuaGQuci5iZXRhIiwiSUdOSVRFX0NUSV9TVVBFUl9BRE1JTiIsIk5TSV9EQVRBX1JFQUQiLCJOU0lfTUVUQURBVEFfUkVBRCIsIk5TSV9GSUxFX0RPV05MT0FEIiwiTlNJX1JQTF9STSIsIk5TSV9SUExfV08iLCJkYXQuY2NtLmhkLnIiLCJWVUxOX0FQSSJdLCJpZCI6IjAwdWI2eXlydWNmZ3lXWnRCNWQ3IiwiZW1haWwiOiJkc3Ryb3VkQGZsYXNocG9pbnQtaW50ZWwuY29tIiwib2t0YV91aWQiOiIwMHViNnl5cnVjZmd5V1p0QjVkNyIsInNpZCI6IjAwMW8wMDAwMDB5UFhYdUFBTyIsIm9pZCI6NTV9.ncnuU98Q9mxRC96cuMZDwYKTWBfrcZ-d9ElJHj_66rlvFNYwtZz9OjYpyXvWr6YqgrF3KRmSZ-oEsU6o9Cukjm964OTwazC2vx94pG_up3eUqBTab7-HXelxALu8p1W2dsZCiUakHDtPa6oFJOglYz-EPReS3Cm3oMBfXEbusKrBHTRQ4y0gyrHQ98WMi0ybpk28m3ffpgSyW8a_juzKl9KLn_yuu-omlLe2DC_82WzCYvkUOXKIbk6rOanxMkWt0kabEj5Yo89keMxIie2utO7yDS0zGfaFa6myZpsvZxett4NznwthpReZwZMY-1wbiFpSODkQxCBH-a3Ba4zU_CPD836_Pt9Hjdbc-6AOrxB946K3MI6rrsAAoFFrJi1TCrhRXXPNWLRAf2NQflks_sNoUdCe8wxYUU-I2T5xi168mocdtFS36q34EcFaXDwCv8G0Hm2bIwHU3rwpqGHYL-XD2JVQiyctEWy6mqw4jicZiSTScgrwRFCJHTyl4E3vVOVvIO4UJRVP7fJQfia2s-gRANiI6J6Ekt4RgRbQS54LKgmAUnyObM9AA5Po6KLQ1r4Zp9rTOsEsEqKpEuLZcI-4vL63hiGDtkQZOpUI4ClNVfJ_x750pMdNl_hcaubKpGCsC-Q92Le5UvgKuYzbHQGxbAoRRgXvrZAS8zqlIo8" "https://api.flashpoint.io/sources/v2/strategic-entities/search?query=test&limit=2"

curl --location 'https://api.flashpoint.io/sources/v2/strategic-entities/chunks/search' \
--header 'Content-Type: application/json' \
--header 'Authorization: eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJpYXQiOjE3NDk3NTI0NzIsImF1ZCI6ImFwaTovL2lnbml0ZSIsImlzcyI6IjAwdWI2eXlydWNmZ3lXWnRCNWQ3OjBvYTlmdHBnc3dTZTBlMDNtNWQ3OnRlc3RlciIsImV4cCI6MzMyODU3NTI0NzIsInZlciI6MSwianRpIjoiQVQuWHdoOG1yNkxTaFJ6YU95eFVYQzlrdDFBZnFEMGQ0T2lmckt0Y2RsZzA3VS5vYXIzZmFxaHN5NG9wUE5iZzVkNyIsImNpZCI6IjBvYTlmdHBnc3dTZTBlMDNtNWQ3IiwidWlkIjoiMDB1YjZ5eXJ1Y2ZneVdadEI1ZDciLCJzY3AiOlsib2ZmbGluZV9hY2Nlc3MiLCJwcm9maWxlIiwiZW1haWwiLCJvcGVuaWQiXSwiYXV0aF90aW1lIjoxNzQ5NzMzODAwLCJzdWIiOiJkc3Ryb3VkQGZsYXNocG9pbnQtaW50ZWwuY29tIiwidXNuIjoiZHN0cm91ZEBmbGFzaHBvaW50LWludGVsLmNvbSIsImZmcyI6W10sInNmX2lkIjoiMDAxbzAwMDAwMHlQWFh1QUFPIiwib3JnX2lkIjo1NSwidXNyX2lkIjoiMDB1YjZ5eXJ1Y2ZneVdadEI1ZDciLCJvZ24iOiJGbGFzaHBvaW50IiwiZ3JvdXBzIjpbIklHTklURS5DQ01DX0hPU1RfQVRUUklCVVRFUy5VU0VSIiwiRUNIT1NFQy5OU0kuVVNFUiIsIklHTklURS5DQ01DX0FQSV9BTkRfVUkuVVNFUiIsIklHTklURS5CUkFORF9JTlRFTC5JTlRFUk5BTCIsIklHTklURS5OU0lfU0VELklOVEVSTkFMIiwiRmxhc2hwb2ludCIsIklHTklURS5CUkFORF9JTlRFTF9QT1YuSU5URVJOQUwiLCJJR05JVEUuU1VQRVIuSU5URVJOQUwiLCJFQ0hPU0VDLlNVUEVSLklOVEVSTkFMIiwiRUNIT1NFQy5IT1NUQkFTRUQuVVNFUiIsIlZVTE5EQl9PUkdfVVNFUiIsIkVDSE9TRUMuQkVBQ09OLlVTRVIiLCJJR05JVEUuRlJBVURfSU5URUwuVVNFUiIsIkV2ZXJ5b25lIiwiRUNIT1NFQy5CQVNJQy5VU0VSIiwiTUZBIE9wdGVkLWluIiwiSUdOSVRFLlZVTE5fQVBJLlVTRVIiLCJJR05JVEUuQ0NNQ19IT1NUX0RBVEEuVVNFUiIsIkVDSE9TRUMuVFdJVFRFUi5VU0VSIiwiUmVhZE1lIEFkbWlucyIsIlRPU19TSUdORUQiLCJJREVOVElGWS5CQVNJQy5VU0VSIiwiSUdOSVRFLkNDTV9IT1NUQkFTRURfQVBJLlVTRVIiLCJFQ0hPU0VDLkFQQUNfREFUQS5VU0VSIiwiSUdOSVRFLkJSQU5EX0lOVEVMLlVTRVIiLCJJR05JVEUuVlVMTl9QUkVNSVVNLlVTRVIiLCJNQU5BR0VEX0FUVFJJQlVUSU9OLkRFVkVMT1BNRU5ULklOVEVSTkFMIiwiRUNIT1NFQy5NRUFfREFUQS5VU0VSIiwiSUdOSVRFLk5TSV9TRUQuVVNFUiIsIklHTklURS5CQVNJQy5VU0VSIiwiTUFOQUdFRF9BVFRSSUJVVElPTi5ERU1PLlRSSUFMIl0sInBybSI6WyJJR05JVEVfVUkiLCJJR05JVEVfQVBJIiwiQ1RJX1JGSV9DUlVEIiwiQ1RJX1JGSV9BRE1JTiIsIkNUSV9HRU5fQ09MTEVDVCIsIkNUSV9SQU5TT01XQVJFIiwiVlVMTl9QUk9EVUNUU19PUkctVVNFUiIsIlZVTE5fVkVORE9SU19PUkctVVNFUiIsIlZVTE5fVlVMTkVSQUJJTElUSUVTX09SRy1VU0VSIiwiVlVMTl9NRU5USU9OU19PUkctVVNFUiIsIlZVTE5fQUxFUlQtUlVMRVNfT1JHLVVTRVIiLCJWVUxOX0FMRVJUU19PUkctVVNFUiIsImRhdC5lZG0uaGQub3JnLnIiLCJkYXQuZWRtLm9yZy5yIiwiZGF0LmluZC5yIiwiZGF0Lm1lZC5kIiwiZGF0Lm1rdC5yIiwiVlVMTl9DVElfUFJFTUlVTSIsIlZVTE5fQ1RJX0JBU0lDIiwiVlVMTl9BTEVSVC1SVUxFU19TVVBFUi1BRE1JTiIsIlZVTE5fQUxFUlRTX1NVUEVSLUFETUlOIiwiVlVMTl9BTEVSVC1URU1QTEFURVNfU1VQRVItQURNSU4iLCJJR05JVEVfQ1RJX1JFUE9SVFMiLCJJR05JVEVfUFNJX1JFUE9SVFMiLCJJR05JVEVfVlVMTl9SRVBPUlRTIiwiSUdOSVRFX0ZSQVVEX1JFUE9SVFMiLCJJR05JVEVfTkFUU0VDX1JFUE9SVFMiLCJkYXQuZWRtLmhkLnIiLCJkYXQuZWRtLm9yZy53IiwiZGF0Lmdscy5yIiwiZGF0LmVkbS5yIiwiZGF0LnJlcC5hc3MuciIsIklHTklURV9BTEVSVElOR19BRE1JTiIsImRhdC51c3IuciIsImRhdC51c3IudyIsIkNUSV9BU1NFVF9DUlVEIiwiQ1RJX0FTU0VUX1IiLCJkYXQuZG0uciIsImRhdC5kbS53IiwiZGF0LmRtLmFkbS5yIiwiZGF0LmRtLmFkbS53IiwiZGF0LmRtLnBvdi53IiwiZGF0LnJlcC53IiwiZGF0LnJlcC5yIiwiZGF0LnJlcC5hc3MudyIsImRhdC50b3AudyIsImRhdC5lZG0udyIsIkZJUkVIT1NFX0RFRkFVTFRfREFUQSIsIkZJUkVIT1NFX1NFTlNJVElWRV9EQVRBIiwiZGF0LmNjbS5jdXMuciIsImRhdC5jY20uY3VzLmhkLnIiLCJkYXQuZGVhLnIiLCJkYXQuZXAuY3JlZC5yIiwiZGF0LmVwLmNyZWQudyIsImRhdC5jc2IuciIsImRhdC5jZm0uYmEuciIsImRhdC5jZm0uciIsImRhdC5jZm0udyIsIkNDTUMuVUkiLCJkYXQuY2NtLmN1cy5oYS5yIiwiZGF0LmNjbS5jdXMuaGEuci5iZXRhIiwiZGF0LmNjbS5jdXMuci5iZXRhIiwiZGF0LmNjbS5jdXMuaGQuci5iZXRhIiwiSUdOSVRFX0NUSV9TVVBFUl9BRE1JTiIsIk5TSV9EQVRBX1JFQUQiLCJOU0lfTUVUQURBVEFfUkVBRCIsIk5TSV9GSUxFX0RPV05MT0FEIiwiTlNJX1JQTF9STSIsIk5TSV9SUExfV08iLCJkYXQuY2NtLmhkLnIiLCJWVUxOX0FQSSJdLCJpZCI6IjAwdWI2eXlydWNmZ3lXWnRCNWQ3IiwiZW1haWwiOiJkc3Ryb3VkQGZsYXNocG9pbnQtaW50ZWwuY29tIiwib2t0YV91aWQiOiIwMHViNnl5cnVjZmd5V1p0QjVkNyIsInNpZCI6IjAwMW8wMDAwMDB5UFhYdUFBTyIsIm9pZCI6NTV9.ncnuU98Q9mxRC96cuMZDwYKTWBfrcZ-d9ElJHj_66rlvFNYwtZz9OjYpyXvWr6YqgrF3KRmSZ-oEsU6o9Cukjm964OTwazC2vx94pG_up3eUqBTab7-HXelxALu8p1W2dsZCiUakHDtPa6oFJOglYz-EPReS3Cm3oMBfXEbusKrBHTRQ4y0gyrHQ98WMi0ybpk28m3ffpgSyW8a_juzKl9KLn_yuu-omlLe2DC_82WzCYvkUOXKIbk6rOanxMkWt0kabEj5Yo89keMxIie2utO7yDS0zGfaFa6myZpsvZxett4NznwthpReZwZMY-1wbiFpSODkQxCBH-a3Ba4zU_CPD836_Pt9Hjdbc-6AOrxB946K3MI6rrsAAoFFrJi1TCrhRXXPNWLRAf2NQflks_sNoUdCe8wxYUU-I2T5xi168mocdtFS36q34EcFaXDwCv8G0Hm2bIwHU3rwpqGHYL-XD2JVQiyctEWy6mqw4jicZiSTScgrwRFCJHTyl4E3vVOVvIO4UJRVP7fJQfia2s-gRANiI6J6Ekt4RgRbQS54LKgmAUnyObM9AA5Po6KLQ1r4Zp9rTOsEsEqKpEuLZcI-4vL63hiGDtkQZOpUI4ClNVfJ_x750pMdNl_hcaubKpGCsC-Q92Le5UvgKuYzbHQGxbAoRRgXvrZAS8zqlIo8' \
--data '{"query":"david.m.stroud","page":0,"size":10}'
"""

# --- Main Agent Class ---
class OrchestratorAgent:
    """The main agent that plans, routes to tools/agents, and generates responses."""
    
    def __init__(self):
        self.llm = genai.GenerativeModel('gemini-1.5-flash')
        self.web_search_tool = WebSearchTool()
        try:
            SED_API_KEY = os.getenv("SED_API_KEY", "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJpYXQiOjE3NDk3MzM4MDMsImF1ZCI6ImFwaTovL2lnbml0ZSIsImlzcyI6IjAwdWI2eXlydWNmZ3lXWnRCNWQ3OjBvYTlmdHBnc3dTZTBlMDNtNWQ3OmFnZW50IHNtaXRoIiwiZXhwIjozMzI4NTczMzgwMywidmVyIjoxLCJqdGkiOiJBVC5XYzZ6Qk1qdzJWMEdsNVdhX1poTkFzNm51YzNwOVRBUWFmYkdsZEF1WEM4Lm9hcjNmYXFoc3k0b3BQTmJnNWQ3IiwiY2lkIjoiMG9hOWZ0cGdzd1NlMGUwM201ZDciLCJ1aWQiOiIwMHViNnl5cnVjZmd5V1p0QjVkNyIsInNjcCI6WyJvZmZsaW5lX2FjY2VzcyIsInByb2ZpbGUiLCJlbWFpbCIsIm9wZW5pZCJdLCJhdXRoX3RpbWUiOjE3NDk3MzM4MDAsInN1YiI6ImRzdHJvdWRAZmxhc2hwb2ludC1pbnRlbC5jb20iLCJ1c24iOiJkc3Ryb3VkQGZsYXNocG9pbnQtaW50ZWwuY29tIiwiZmZzIjpbXSwic2ZfaWQiOiIwMDFvMDAwMDAweVBYWHVBQU8iLCJvcmdfaWQiOjU1LCJ1c3JfaWQiOiIwMHViNnl5cnVjZmd5V1p0QjVkNyIsIm9nbiI6IkZsYXNocG9pbnQiLCJncm91cHMiOlsiSUdOSVRFLkNDTUNfSE9TVF9BVFRSSUJVVEVTLlVTRVIiLCJFQ0hPU0VDLk5TSS5VU0VSIiwiSUdOSVRFLkNDTUNfQVBJX0FORF9VSS5VU0VSIiwiSUdOSVRFLkJSQU5EX0lOVEVMLklOVEVSTkFMIiwiSUdOSVRFLk5TSV9TRUQuSU5URVJOQUwiLCJGbGFzaHBvaW50IiwiSUdOSVRFLkJSQU5EX0lOVEVMX1BPVi5JTlRFUk5BTCIsIklHTklURS5TVVBFUi5JTlRFUk5BTCIsIkVDSE9TRUMuU1VQRVIuSU5URVJOQUwiLCJFQ0hPU0VDLkhPU1RCQVNFRC5VU0VSIiwiVlVMTkRCX09SR19VU0VSIiwiRUNIT1NFQy5CRUFDT04uVVNFUiIsIklHTklURS5GUkFVRF9JTlRFTC5VU0VSIiwiRXZlcnlvbmUiLCJFQ0hPU0VDLkJBU0lDLlVTRVIiLCJNRkEgT3B0ZWQtaW4iLCJJR05JVEUuVlVMTl9BUEkuVVNFUiIsIklHTklURS5DQ01DX0hPU1RfREFUQS5VU0VSIiwiRUNIT1NFQy5UV0lUVEVSLlVTRVIiLCJSZWFkTWUgQWRtaW5zIiwiVE9TX1NJR05FRCIsIklERU5USUZZLkJBU0lDLlVTRVIiLCJJR05JVEUuQ0NNX0hPU1RCQVNFRF9BUEkuVVNFUiIsIkVDSE9TRUMuQVBBQ19EQVRBLlVTRVIiLCJJR05JVEUuQlJBTkRfSU5URUwuVVNFUiIsIklHTklURS5WVUxOX1BSRU1JVU0uVVNFUiIsIk1BTkFHRURfQVRUUklCVVRJT04uREVWRUxPUE1FTlQuSU5URVJOQUwiLCJFQ0hPU0VDLk1FQV9EQVRBLlVTRVIiLCJJR05JVEUuTlNJX1NFRC5VU0VSIiwiSUdOSVRFLkJBU0lDLlVTRVIiLCJNQU5BR0VEX0FUVFJJQlVUSU9OLkRFTU8uVFJJQUwiXSwicHJtIjpbIklHTklURV9VSSIsIklHTklURV9BUEkiLCJDVElfUkZJX0NSVUQiLCJDVElfUkZJX0FETUlOIiwiQ1RJX0dFTl9DT0xMRUNUIiwiQ1RJX1JBTlNPTVdBUkUiLCJWVUxOX1BST0RVQ1RTX09SRy1VU0VSIiwiVlVMTl9WRU5ET1JTX09SRy1VU0VSIiwiVlVMTl9WVUxORVJBQklMSVRJRVNfT1JHLVVTRVIiLCJWVUxOX01FTlRJT05TX09SRy1VU0VSIiwiVlVMTl9BTEVSVC1SVUxFU19PUkctVVNFUiIsIlZVTE5fQUxFUlRTX09SRy1VU0VSIiwiZGF0LmVkbS5oZC5vcmcuciIsImRhdC5lZG0ub3JnLnIiLCJkYXQuaW5kLnIiLCJkYXQubWVkLmQiLCJkYXQubWt0LnIiLCJWVUxOX0NUSV9QUkVNSVVNIiwiVlVMTl9DVElfQkFTSUMiLCJWVUxOX0FMRVJULVJVTEVTX1NVUEVSLUFETUlOIiwiVlVMTl9BTEVSVFNfU1VQRVItQURNSU4iLCJWVUxOX0FMRVJULVRFTVBMQVRFU19TVVBFUi1BRE1JTiIsIklHTklURV9DVElfUkVQT1JUUyIsIklHTklURV9QU0lfUkVQT1JUUyIsIklHTklURV9WVUxOX1JFUE9SVFMiLCJJR05JVEVfRlJBVURfUkVQT1JUUyIsIklHTklURV9OQVRTRUNfUkVQT1JUUyIsImRhdC5lZG0uaGQuciIsImRhdC5lZG0ub3JnLnciLCJkYXQuZ2xzLnIiLCJkYXQuZWRtLnIiLCJkYXQucmVwLmFzcy5yIiwiSUdOSVRFX0FMRVJUSU5HX0FETUlOIiwiZGF0LnVzci5yIiwiZGF0LnVzci53IiwiQ1RJX0FTU0VUX0NSVUQiLCJDVElfQVNTRVRfUiIsImRhdC5kbS5yIiwiZGF0LmRtLnciLCJkYXQuZG0uYWRtLnIiLCJkYXQuZG0uYWRtLnciLCJkYXQuZG0ucG92LnciLCJkYXQucmVwLnciLCJkYXQucmVwLnIiLCJkYXQucmVwLmFzcy53IiwiZGF0LnRvcC53IiwiZGF0LmVkbS53IiwiRklSRUhPU0VfREVGQVVMVF9EQVRBIiwiRklSRUhPU0VfU0VOU0lUSVZFX0RBVEEiLCJkYXQuY2NtLmN1cy5yIiwiZGF0LmNjbS5jdXMuaGQuciIsImRhdC5kZWEuciIsImRhdC5lcC5jcmVkLnIiLCJkYXQuZXAuY3JlZC53IiwiZGF0LmNzYi5yIiwiZGF0LmNmbS5iYS5yIiwiZGF0LmNmbS5yIiwiZGF0LmNmbS53IiwiQ0NNQy5VSSIsImRhdC5jY20uY3VzLmhhLnIiLCJkYXQuY2NtLmN1cy5oYS5yLmJldGEiLCJkYXQuY2NtLmN1cy5yLmJldGEiLCJkYXQuY2NtLmN1cy5oZC5yLmJldGEiLCJJR05JVEVfQ1RJX1NVUEVSX0FETUlOIiwiTlNJX0RBVEFfUkVBRCIsIk5TSV9NRVRBREFUQV9SRUFEIiwiTlNJX0ZJTEVfRE9XTkxPQUQiLCJOU0lfUlBMX1JNIiwiTlNJX1JQTF9XTyIsImRhdC5jY20uaGQuciIsIlZVTE5fQVBJIl0sImlkIjoiMDB1YjZ5eXJ1Y2ZneVdadEI1ZDciLCJlbWFpbCI6ImRzdHJvdWRAZmxhc2hwb2ludC1pbnRlbC5jb20iLCJva3RhX3VpZCI6IjAwdWI2eXlydWNmZ3lXWnRCNWQ3Iiwic2lkIjoiMDAxbzAwMDAwMHlQWFh1QUFPIiwib2lkIjo1NX0.j9MXGO0Ax_GkEOx-ovnJkhivP-YzsC57WUrSj8F1ZxjHnK07kpQU4iXi49e8IqlMvwxjtkoWuvLoP1XISFljg0A1IZf3ZrGsvkzTkqblUi-DhlLwKB64jpGfGxzx9J6EfSHMaTynuD1O-UkqLLbf_htxSE8LJp-i4NYlr_ubtiq2S_VxbpgTnx6U3gyXpGue2hyHsFXJug0l3lHzbTCJRatnsCccC73uEucy_kIf2MBkNuoJ18y6J8www8MrDDHBrRoTo-qYk5KdV-_oe6nmVsA4M2MrLIKhBWYnXY4jsf9IPYlOf8CxuKZOxbYG8B_X0pn_SRvZhjwxodlfRNIx9aDYVejbPj-NoZhW7381lFeR17MdJ3kOQiS9m_kd5PC9gea-Ek1aJYBSSqtSEtVa7kfuMnCySv9afx6p4w3kOa5UEoCih9xChUP057VtAJ1vhqZbf3AeS-Cae7I8zECK-Sz1seITXFXOfzHPIv7OGyQmDgkK-o1RSSEX7SNtvypDSpS7xL9WkQ3EXRfBv9wbbKO5W8LB4Dfhb9YAhAxsQNQ0MuvubllsGPOv5iU1o--4aLJpe3-rm8kcXooi4Sa-m42KL_1FnmWo7p9P8nmZiMqR7XGhnL2idaSi7ktPZyTr8QyDsbMVsqxHLht8THVjjzFlv0wC2_O3a4-R0nIoSqM")
            SED_API_BASE_URL = "https://api.flashpoint.io/sources/v2/strategic-entities"
            sed_search_tool = SEDSearchTool(api_key=SED_API_KEY, base_url=SED_API_BASE_URL)
            self.sed_agent = SEDAgent(llm=self.llm, sed_tool=sed_search_tool)
        except ValueError as e:
            logger.error(e)
            self.sed_agent = None
        self.graph = self._initialize_graph()
        
    def _initialize_graph(self) -> StateGraph:
        """Initializes the main agent's workflow graph."""
        graph = StateGraph(MainAgentState)
        
        graph.add_node("plan_and_route", self._plan_and_route)
        graph.add_node("perform_web_search", self._perform_web_search)
        graph.add_node("call_sed_agent", self._call_sed_agent)
        graph.add_node("generate_response", self._generate_response)
        
        graph.set_entry_point("plan_and_route")
        
        graph.add_conditional_edges(
            "plan_and_route",
            lambda state: state["tool_choice"],
            {
                "web_search": "perform_web_search",
                "sed_search": "call_sed_agent",
                "none": "generate_response"
            }
        )
        
        graph.add_edge("perform_web_search", "generate_response")
        graph.add_edge("call_sed_agent", "generate_response")
        graph.add_edge("generate_response", END)
        
        return graph.compile()

    def _plan_and_route(self, state: MainAgentState) -> MainAgentState:
        json_llm = genai.GenerativeModel('gemini-1.5-flash', generation_config={"response_mime_type": "application/json"})
        conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in state["messages"]])
        latest_question = state["messages"][-1]["content"]

        prompt = f"""You are an expert planner for an AI assistant. You have two tools:
1.  `web_search`: For general knowledge, current events, and public information.
2.  `sed_search`: For a deeper investigation into people and organizations.

Analyze the user's latest question in the context of the conversation and choose the single best tool to answer it.

**Conversation History:**
{conversation_history}

**User's Latest Question:** "{latest_question}"

**Instructions:**
1.  **Analyze Intent:** Determine if the user is asking about a public topic or an internal one.
2.  **Rewrite for Clarity:** Rewrite the question to be a self-contained query.
3.  **Choose Tool:** Select "sed_search" for internal topics, "web_search" for public topics, or "none" for greetings/chitchat.
4.  **Output:** Respond in a JSON object with "tool_choice" (string) and "query" (string).
"""
        if not self.sed_agent:
            prompt += "\n**Note:** The `sed_search` tool is currently unavailable. Do not choose it."
        try:
            response = json_llm.generate_content(prompt)
            plan = json.loads(response.text)
            tool_choice = plan.get("tool_choice", "none")
            if tool_choice == "sed_search" and not self.sed_agent:
                tool_choice = "web_search"
            
            state["tool_choice"] = tool_choice
            state["search_query"] = plan.get("query", "")
            logger.info(f"Planner decision: Tool = {state['tool_choice']}, Query = '{state['search_query']}'")
        except Exception as e:
            logger.error(f"Failed to parse planning response: {e}. Defaulting to no tool.")
            state["tool_choice"] = "none"
            state["search_query"] = ""
        return state

    def _perform_web_search(self, state: MainAgentState) -> MainAgentState:
        query = state["search_query"]
        state["search_results"] = self.web_search_tool.run(query)
        return state

    def _call_sed_agent(self, state: MainAgentState) -> MainAgentState:
        logger.info("Orchestrator: Calling SEDAgent.")
        query = state["search_query"]
        state["sed_summary"] = self.sed_agent.run(query)
        return state

    def _generate_response(self, state: MainAgentState) -> MainAgentState:
        web_results = state.get("search_results")
        sed_summary = state.get("sed_summary")
        conversation = "\n".join([f"{msg['role']}: {msg['content']}" for msg in state["messages"]])
        
        prompt_template = f"You are Agent Smith, a helpful AI assistant. Answer the user's last question based on the conversation and any provided search results.\n\n**Conversation History:**\n{conversation}"

        if web_results:
            formatted_results = "\n\n".join([f"Title: {r['title']}\nSnippet: {r['snippet']}\nURL: {r['url']}" for r in web_results])
            prompt_template += f"\n\n**Web Search Results:**\n<web_search_results>\n{formatted_results}\n</web_search_results>\n\nSynthesize an answer from the web results. Cite URLs."
        elif sed_summary:
            prompt_template += f"\n\n**Internal Document Summary (from SED Agent):**\n<sed_summary>\n{sed_summary}\n</sed_summary>\n\nSynthesize an answer based on the internal document summary."
        else:
            prompt_template += "\n\nProvide a conversational response based on the history."
        
        logger.info("Generating final response...")
        response = self.llm.generate_content(prompt_template)
        state["messages"].append({"role": "assistant", "content": response.text})
        return state

    def process_message(self, messages: List[Dict[str, str]]) -> Dict:
        initial_state = MainAgentState(
            messages=messages, tool_choice="none", search_query="",
            search_results=[], sed_summary=""
        )
        try:
            final_state = self.graph.invoke(initial_state)
            return final_state
        except Exception as e:
            logger.error(f"Error in graph invocation: {e}", exc_info=True)
            return {"messages": messages + [{"role": "assistant", "content": f"I'm sorry, an error occurred: {e}"}]}
