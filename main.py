from flask import Flask, request, jsonify, make_response  # Import make_response
from flask_cors import CORS
import json
import requests
import re
import os

app = Flask(__name__)
CORS(app)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_DEFAULT_MODEL = "llama3-8b-8192"  # Or your preferred model


# --- Helper Functions ---

# --- Text Normalization for Exact Comparison ---
def normalize_for_exact_comparison(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) 
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_party_names_with_source_from_case(case_dict: dict) -> list[tuple[str, str, str]]:
    names_with_source = []
    processed_normalized_names = set()

    def add_name_with_source(raw_name_str, source_desc):
        if isinstance(raw_name_str, str) and raw_name_str.strip():
            norm_name = normalize_for_exact_comparison(raw_name_str)
            if norm_name and norm_name not in processed_normalized_names:
                names_with_source.append((raw_name_str.strip(), norm_name, source_desc))
                processed_normalized_names.add(norm_name)

    if isinstance(case_dict.get("petitioner"), str): add_name_with_source(case_dict["petitioner"], "petitioner")
    if isinstance(case_dict.get("respondent"), str): add_name_with_source(case_dict["respondent"], "respondent")
    
    for party_list_key in ["gfc_petitioners", "gfc_respondents"]:
        value = case_dict.get(party_list_key)
        if isinstance(value, list):
            for i, party_obj in enumerate(value):
                if isinstance(party_obj, dict): add_name_with_source(party_obj.get("name"), f"{party_list_key}[{i}].name")
    
    gfc_orders = case_dict.get("gfc_orders_data", {})
    if isinstance(gfc_orders, dict):
        for party_list_key in ["petitioners", "respondents"]:
            for i, party_obj in enumerate(gfc_orders.get(party_list_key, [])):
                if isinstance(party_obj, dict): add_name_with_source(party_obj.get("name"), f"gfc_orders_data.{party_list_key}[{i}].name")

    raw_resp_addr = str(case_dict.get("respondent_address", ""))
    for line_num, line in enumerate(raw_resp_addr.splitlines()):
        line = line.strip()
        match = re.match(r'^\s*\d+\)\s*([^,]+?)(?=\s*(?:Ro\b|Tq\b|Dist\b|Advocate\b|$|,))', line, re.IGNORECASE)
        if match:
            potential_name = match.group(1).strip()
            if len(potential_name.split()) > 1 and len(potential_name) < 60 and any(c.isalpha() for c in potential_name):
                common_addr_indicators = ['tq', 'dist', 'road', 'street', 'nagar', 'colony', 'ro ']
                if not any(indicator in potential_name.lower() for indicator in common_addr_indicators):
                    add_name_with_source(potential_name, f"respondent_address (line approx {line_num+1}, heuristic)")
        else: # Check if line itself might be a name before "Advocate"
            name_part = line.split(' Advocate')[0].strip()
            if name_part != line and name_part and len(name_part.split()) > 1 and len(name_part) < 60:
                 add_name_with_source(name_part, f"respondent_address (line approx {line_num+1} before 'Advocate', heuristic)")
                
    return names_with_source



def get_addresses_from_case_dict_normalized(case_dict: dict) -> list:
    addresses_set = set()
    def add_address(addr_str):
        if isinstance(addr_str, str) and addr_str.strip():
            norm_addr = normalize_for_exact_comparison(addr_str)
            if norm_addr: 
                addresses_set.add(norm_addr)

    add_address(case_dict.get("petitioner_address"))
    add_address(case_dict.get("respondent_address"))
    
    for party_list_key in ["gfc_petitioners", "gfc_respondents"]:
        for party_obj in case_dict.get(party_list_key, []):
            if isinstance(party_obj, dict): add_address(party_obj.get("address"))
    
    gfc_orders = case_dict.get("gfc_orders_data", {})
    if isinstance(gfc_orders, dict):
        for party_list_key in ["petitioners", "respondents"]:
            for party_obj in gfc_orders.get(party_list_key, []):
                if isinstance(party_obj, dict): add_address(party_obj.get("address"))
    return list(addresses_set)


def compare_request_to_case(request_data: dict, case_dict: dict) -> tuple[bool, list, list]:
    """
    Compares request fields to a single case dictionary.
    Returns: (
        primary_name_matched_in_case, 
        list_of_matched_field_details,  (each dict includes 'matched_in_case_field_source' for names)
        list_of_mismatched_field_details
    )
    """
    primary_name_matched = False
    all_party_names_with_source_in_case = get_party_names_with_source_from_case(case_dict)
    
    matched_fields_details = []
    mismatched_fields_details = []
    
    norm_req_primary_name = ""
    if "name" in request_data: 
        norm_req_primary_name = normalize_for_exact_comparison(str(request_data["name"]))

    if norm_req_primary_name:
        name_found_source_desc = None
        original_matched_name_in_case = norm_req_primary_name 
        for raw_name, norm_name, source_desc in all_party_names_with_source_in_case:
            if norm_name == norm_req_primary_name:
                primary_name_matched = True
                name_found_source_desc = source_desc
                original_matched_name_in_case = raw_name
                break
        
        if primary_name_matched:
            matched_fields_details.append({
                "field": "name", 
                "request_value": request_data['name'], 
                "matched_case_value": original_matched_name_in_case,
                "matched_in_case_field_source": name_found_source_desc 
            })
        else:
            mismatched_fields_details.append({
                "field": "name", 
                "request_value": request_data['name'], 
                "reason": f"Normalized '{norm_req_primary_name}' not found in case party names."
            })
    else:
        mismatched_fields_details.append({"field": "name", "request_value": "", "reason": "'name' field missing or empty in JSON Request."})

    for req_key, req_value_raw in request_data.items():
        if req_key == "name": continue 

        norm_req_val = normalize_for_exact_comparison(str(req_value_raw))
        field_matched_in_this_case = False
        actual_case_value_matched = "N/A"
        match_source_description = None 
        
        if not norm_req_val and not str(req_value_raw).strip(): 
            case_val_for_empty_check_raw = case_dict.get(req_key)
            if case_val_for_empty_check_raw is None or not str(case_val_for_empty_check_raw).strip():
                field_matched_in_this_case = True
                actual_case_value_matched = ""
            else:
                 mismatched_fields_details.append({"field": req_key, "request_value": str(req_value_raw), "reason": f"Request value empty, but case has '{str(case_val_for_empty_check_raw)}'."})
        
        elif req_key == "father_name":
            found_father_name_component = False
            case_party_names_normalized = [item[1] for item in all_party_names_with_source_in_case]
            for i, norm_party_text in enumerate(case_party_names_normalized):
                if norm_req_val in norm_party_text.split(): 
                    field_matched_in_this_case = True
                    raw_name_for_context = all_party_names_with_source_in_case[i][0]
                    source_desc_for_context = all_party_names_with_source_in_case[i][2]
                    actual_case_value_matched = f"Found in party name: '{raw_name_for_context}'"
                    match_source_description = source_desc_for_context 
                    break
            if not field_matched_in_this_case:
                mismatched_fields_details.append({"field": req_key, "request_value": str(req_value_raw), "reason": "Not found as a distinct word in case party names."})
        
        elif req_key == "address":
            case_addresses_normalized = get_addresses_from_case_dict_normalized(case_dict)
            if norm_req_val in case_addresses_normalized:
                field_matched_in_this_case = True
                actual_case_value_matched = f"Matched (normalized): '{norm_req_val}'" 
            else:
                mismatched_fields_details.append({"field": req_key, "request_value": str(req_value_raw), "reason": f"Normalized address not in case addresses." })
        
        else: 
            case_val_raw = case_dict.get(req_key, "") 
            case_val_norm = normalize_for_exact_comparison(str(case_val_raw))
            if norm_req_val == case_val_norm:
                field_matched_in_this_case = True
                actual_case_value_matched = str(case_val_raw) 
            else:
                mismatched_fields_details.append({"field": req_key, "request_value": str(req_value_raw), "reason": f"Normalized '{norm_req_val}' != Case normalized '{case_val_norm}' (Original case value: '{str(case_val_raw)}')."})

        if field_matched_in_this_case:
            match_detail = {
                "field": req_key, 
                "request_value": str(req_value_raw),
                "matched_case_value": actual_case_value_matched
            }
            if match_source_description: 
                match_detail["matched_in_case_field_source"] = match_source_description
            matched_fields_details.append(match_detail)
            
    return primary_name_matched, matched_fields_details, mismatched_fields_details


def clean_llm_output(text: str) -> str:
    if not text: return ""
    cleaned = re.sub(r"^\s*[\*\-]\s+", "", text.strip())
    return cleaned




# --- Core Logic ---

def analyze_cases(request_data, response_data):
    """
    Core analysis logic with improved error handling and validation.
    """
    try:
        # --- Validations (Raise exceptions for errors) ---
        if not request_data or not response_data:
            raise ValueError("Both request and response data are required.")

        if not (isinstance(request_data, dict) and request_data and "name" in request_data and request_data["name"].strip()):
            raise ValueError("Invalid request_data. Must be non-empty with a 'name'.") # Raise ValueError

        if not (isinstance(response_data.get("data"), dict) and isinstance(response_data.get("data", {}).get("result"), list)):
            raise ValueError("Invalid response_data structure. Expected 'data.result' list.") # Raise ValueError
        

        analysis_results = []
        cases = response_data.get("data", {}).get("result", [])
        if not cases:  # This is fine - no cases to process
            return []

        for index, case_item_dict in enumerate(cases):
            if not isinstance(case_item_dict, dict):
                continue  # Skip invalid case items

            try:  # Inner try-except for individual case processing
                primary_name_matched, matched_fields_in_case, mismatched_fields_in_case = \
                    compare_request_to_case(request_data, case_item_dict)
        
                is_potentially_relevant = False
                if matched_fields_in_case:
                    non_state_matches_exist = any(mf["field"] != "state" for mf in matched_fields_in_case)
                    if non_state_matches_exist:
                        is_potentially_relevant = True
        
                potential_relevance_status_for_case = "Potentially Relevant" if is_potentially_relevant else "Potentially Not Relevant"
            
                case_identifiers_for_llm = {
                    "case_name": str(case_item_dict.get("case_name", "N/A"))[:70],
                    "year": case_item_dict.get("year", "N/A"),
                    "court_name": str(case_item_dict.get("court_name", "N/A"))[:50],
                    "petitioner": str(case_item_dict.get("petitioner", "N/A"))[:40],
                    "respondent": str(case_item_dict.get("respondent", "N/A"))[:40],
                    "case_link": case_item_dict.get("case_link") or case_item_dict.get("case_details_link"),
                    "case_status": case_item_dict.get("case_status", "N/A"),
                    "case_type_name": case_item_dict.get("case_type_name", "N/A")
                }
            
                llm_matched_fields_payload = []
                for mf in matched_fields_in_case:
                    detail = {
                        "request_field": mf["field"],
                        "request_value": str(mf["request_value"])[:50],
                        "matched_value_in_case": str(mf["matched_case_value"])[:50]
                    }
                    if mf.get('matched_in_case_field_source'):
                        detail["source_in_case"] = str(mf['matched_in_case_field_source'])[:50]
                    llm_matched_fields_payload.append(detail)
        
                llm_mismatched_fields_payload = [
                    {
                        "request_field": mm["field"],
                        "request_value": str(mm["request_value"])[:50],
                        "reason": mm["reason"][:70]
                    }
                    for mm in mismatched_fields_in_case
                ]
        
                llm_prompt_input_summary = {
                    "case_identifiers": case_identifiers_for_llm,
                    "primary_name_match_status_for_case": "Matched" if primary_name_matched else "Not Matched",
                    "overall_potential_relevance_for_case": potential_relevance_status_for_case,
                    "details_of_matched_fields": llm_matched_fields_payload,
                    "details_of_mismatched_fields": llm_mismatched_fields_payload,
                }
        
                matched_bullets_str = ""
                if llm_prompt_input_summary['details_of_matched_fields']:
                    for item in llm_prompt_input_summary['details_of_matched_fields']:
                        source_info = f" (found in case field: {item.get('source_in_case', 'Direct field match')})" if item.get('source_in_case') else ""
                        matched_bullets_str += f"    *   The requested '{item['request_field']}' ('{item['request_value']}') matched the case's '{item['matched_value_in_case']}'{source_info}.\n"
                else:
                    matched_bullets_str = "    *   No data points from the request found an exact match in this case.\n"
        
                mismatched_bullets_str = ""
                if llm_prompt_input_summary['details_of_mismatched_fields']:
                    for item in llm_prompt_input_summary['details_of_mismatched_fields']:
                        mismatched_bullets_str += f"    *   The requested '{item['request_field']}' ('{item['request_value']}') did not find an exact match (Reason: {item['reason']}).\n"
                else:
                    mismatched_bullets_str = "    *   All other request fields (if any) either matched or were not applicable for mismatch reporting.\n"
        
                # --- REFINED REASON FOR RELEVANCE TEXT LOGIC ---
                reason_for_relevance_text_for_llm = ""
                if llm_prompt_input_summary['overall_potential_relevance_for_case'] == "Potentially Relevant":
                    # This means non_state_matches_exist was True in the Python logic.
                    # We count only these non-state matches from the payload for the explanation.
                    actual_significant_matches_payload = [
                        mf_payload for mf_payload in llm_prompt_input_summary['details_of_matched_fields']
                        if mf_payload['request_field'] != 'state'
                    ]
                    count_significant_matches = len(actual_significant_matches_payload)
        
                    if count_significant_matches > 0:
                        reason_for_relevance_text_for_llm = (
                            f"This case is flagged as potentially relevant because {count_significant_matches} "
                            f"significant data point(s) (other than just 'state') from the request were found in this case record, "
                            f"as detailed in the 'Matched Data Points' section. The significance of these matches suggests a potential connection."
                        )
                    else:
                        # This path indicates an inconsistency: Python logic determined 'Potentially Relevant' (implying non-state matches),
                        # but no non-state matches were found in the llm_matched_fields_payload.
                        # This could happen if `matched_fields_in_case` (from compare_request_to_case) had non-state matches,
                        # but they were somehow lost or not correctly translated into `llm_matched_fields_payload`.
                        # Or, it could be a very unusual edge case in data.
                        reason_for_relevance_text_for_llm = (
                            "This case is flagged as potentially relevant based on initial criteria. "
                            "However, there's a discrepancy in identifying the specific significant non-state matches in the summary. "
                            "Please review the 'Matched Data Points' section carefully. "
                            "(If this issue persists, it might indicate an internal data processing inconsistency for this case)."
                        )
                else: # overall_potential_relevance_for_case == "No Apparent Relevance"
                    if llm_prompt_input_summary['details_of_matched_fields']:
                        # Some matches exist, but not enough for "Potentially Relevant" status by Python logic.
                        # Check if it was a 'state'-only match.
                        is_only_state_match = all(
                            mf_payload['request_field'] == 'state' for mf_payload in llm_prompt_input_summary['details_of_matched_fields']
                        ) and any( # ensure there actually IS a state match
                            mf_payload['request_field'] == 'state' for mf_payload in llm_prompt_input_summary['details_of_matched_fields']
                        )
        
                        if is_only_state_match:
                            reason_for_relevance_text_for_llm = (
                                "This case is flagged as having Potentially Not Relevant because, although the 'state' field matched, "
                                "no other significant data points from the request were found in this case record. "
                                "A 'state'-only match is not considered a strong indicator of specific relevance by the system."
                            )
                        else:
                            # Matches exist, but they are not 'state'-only, yet Python still deemed it "No Apparent Relevance".
                            # This means `non_state_matches_exist` was False, which implies all matches in `matched_fields_in_case`
                            # were 'state', or `matched_fields_in_case` was empty.
                            # The `llm_prompt_input_summary['details_of_matched_fields']` existing here and not being state-only
                            # while relevance is "No Apparent Relevance" is an inconsistency.
                            # However, the most direct explanation for "No Apparent Relevance" when some matches are listed is:
                            reason_for_relevance_text_for_llm = (
                                "This case is flagged as having Potentially Not Relevant. While some data points might have matched "
                                "(as shown in 'Matched Data Points'), they did not meet the criteria for a strong potential connection "
                                "(e.g., a significant non-'state' field match was required by the system but not found or not deemed sufficient)."
                            )
                    else: # No matches at all in llm_prompt_input_summary['details_of_matched_fields']
                        reason_for_relevance_text_for_llm = (
                            "This case is flagged as having Potentially Not Relevant because no data points from the "
                            "request were found to match in this case record."
                        )
                # --- END OF REFINED REASON ---


                 # *** START OF CHANGE ***
                case_link_url = case_identifiers_for_llm.get('case_link')
                llm_case_link_markdown = "Not available" # Default
                if case_link_url and isinstance(case_link_url, str) and \
                   (case_link_url.startswith("http://") or case_link_url.startswith("https://")):
                    llm_case_link_markdown = f"[View Case Details]({case_link_url})"
                elif case_link_url and isinstance(case_link_url, str): # It's a string, but not a typical http/https URL
                    llm_case_link_markdown = f"{case_link_url} (Link provided as text)"
                # *** END OF CHANGE ***


        
                prompt_for_case_explanation = f"""
        You are an AI legal analyst. Your task is to provide a neat, clean, and clear explanation in bullet points about the potential relevance of a specific court case to a person of interest, based on programmatic findings of data overlaps. Do NOT use the word "accused". Do not mention "programmatic" or "normalized" in your output.
        
        Person of Interest Details (from JSON Request):
        - Name: '{str(request_data.get("name"))}'
        - Father's Name: '{str(request_data.get("father_name", "N/A"))}'
        - Address: '{str(request_data.get("address", "N/A"))[:100]}...'
        - Year: '{str(request_data.get("year", "N/A"))}'
        - State: '{str(request_data.get("state", "N/A"))}'
        (Other request fields: {json.dumps({k: str(v)[:30] for k, v in request_data.items() if k not in ['name', 'father_name', 'address', 'year', 'state']}, indent=2)})
        
        Comparison Findings for Case {index + 1} (Summary):
        - Case Identifiers: {json.dumps(llm_prompt_input_summary['case_identifiers'], indent=4)}
        - Primary Name Match Status: {llm_prompt_input_summary['primary_name_match_status_for_case']}
        - Overall Potential Relevance: {llm_prompt_input_summary['overall_potential_relevance_for_case']}
        - Matched Fields Count: {len(llm_prompt_input_summary['details_of_matched_fields'])}
        - Mismatched Fields Count: {len(llm_prompt_input_summary['details_of_mismatched_fields'])}
        
        Explanation for Case {index + 1} (Strictly use bullet points for each section below. Be factual and clear.):
        
        *   Primary Name Assessment:
            *   The name '{request_data.get('name')}' provided in the request was **{llm_prompt_input_summary['primary_name_match_status_for_case']}** with a party name in this case.
            {(f"    *   It matched with: '{llm_prompt_input_summary['details_of_matched_fields'][0]['matched_value_in_case']}' (found in case field: {llm_prompt_input_summary['details_of_matched_fields'][0].get('source_in_case', 'N/A')}).") if primary_name_matched and llm_prompt_input_summary['details_of_matched_fields'] and llm_prompt_input_summary['details_of_matched_fields'][0]['request_field'] == 'name' else ""}
        
        *   Matched Data Points from Request in this Case:
        {matched_bullets_str}
        *   Reason for Potential Relevance ({llm_prompt_input_summary['overall_potential_relevance_for_case']}):
            *   {reason_for_relevance_text_for_llm}
        
        *   Case Description:
            *   Case: {case_identifiers_for_llm['case_name']} ({case_identifiers_for_llm['year']})
            *   Court: {case_identifiers_for_llm['court_name']}
            *   Type: {case_identifiers_for_llm['case_type_name']}
            *   Status: {case_identifiers_for_llm['case_status']}
            *   Petitioner(s): {case_identifiers_for_llm['petitioner']}
            *   Respondent(s): {case_identifiers_for_llm['respondent']}
        
        *   Case Link:
            *   Link: {llm_case_link_markdown}
        
        *   Mismatched Data Points from Request in this Case (for context):
        {mismatched_bullets_str}
        """
                analysis_results.append({
                    "record_source": f"Case ({index + 1})",
                    "primary_name_matched_status": "Yes" if primary_name_matched else "No",
                    "overall_relevance": potential_relevance_status_for_case,
                    "llm_explanation_prompt_messages": [{"role": "user", "content": prompt_for_case_explanation}],
                    "case_details_for_display": case_item_dict,
                })

            except Exception as inner_e:  # Catch exceptions during individual case processing
                # Instead of print:
                app.logger.error(f"Error processing case {index + 1}: {inner_e}", exc_info=True)
                # Choose one of the following options for error handling:
                # 1. Add error information to the results:
                analysis_results.append({
                    "record_source": f"Case ({index + 1})",
                    "error": str(inner_e) 
                })
                # 2. OR, if you want to halt processing on ANY error:
                # raise inner_e  # Re-raise the exception

        return analysis_results
    
    except ValueError as ve:  # Catch validation errors
        raise ve  # Re-raise ValueError for the API endpoint to handle as a 400 error
    except Exception as e: # Catch any other unexpected errors
        raise Exception(f"An unexpected error occurred during analysis: {str(e)}")
    
    


# --- GROQ API Interaction (NO CHANGES) ---


def query_groq_chat(messages, model_name=GROQ_DEFAULT_MODEL, temperature=0.2, max_tokens=1000):
    """Queries the GROQ API for LLM response.  Raises exceptions for errors."""

    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")

    headers = {"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"}
    payload = {"model": model_name, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "stream": False}

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=180)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        response_data = response.json()
        content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return clean_llm_output(content)  # Assuming you have the clean_llm_output function defined

    except requests.exceptions.RequestException as e:
        # More informative error messages, but avoid exposing the full response if it contains sensitive info.
        if e.response is not None and 400 <= e.response.status_code < 500: # Client errors
            raise ValueError(f"GROQ API Client Error ({e.response.status_code}): Check your request data.") # Generic client error
        elif e.response is not None and 500 <= e.response.status_code: # Server errors
            raise Exception(f"GROQ API Server Error ({e.response.status_code}): Please try again later.") # Generic server error
        else:
            raise Exception(f"Error communicating with GROQ API: {e}")  # Network or other issues

    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"Unexpected response format from GROQ API: {e}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred during GROQ interaction: {str(e)}")  # Catch-all





# --- API Endpoints ---
@app.route('/analyze_cases', methods=['POST'])
def analyze_cases_api():
    try:
        data = request.get_json()
        request_data = data.get('request_data')
        response_data = data.get('response_data')

        analysis_results = analyze_cases(request_data, response_data)  # Call the combined analysis function
        return jsonify(analysis_results)

    except ValueError as ve: # Catches validation errors from analyze_cases or data extraction
        app.logger.warning(f"Bad request to /analyze_cases: {ve}")
        return make_response(jsonify({"error": str(ve)}), 400)
    except Exception as e:
        app.logger.error(f"Internal server error in /analyze_cases: {e}", exc_info=True)
        return make_response(jsonify({"error": "An internal server error occurred."}), 500)
    


@app.route('/get_llm_explanation', methods=['POST'])
def get_llm_explanation_api():
    try:
        if not request.is_json:
            return make_response(jsonify({"error": "Request body must be JSON."}), 415)
        data = request.get_json()
        if data is None:
            return make_response(jsonify({"error": "Invalid JSON payload."}), 400)

        prompt_messages = data.get('prompt_messages')

        if not prompt_messages:
            raise ValueError("`prompt_messages` is required in the request body.")
        if not isinstance(prompt_messages, list): # Already good
            raise TypeError("`prompt_messages` must be a list of dictionaries.")
        # Add a check for the content of the list items
        if not all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in prompt_messages):
            raise ValueError("Each item in `prompt_messages` must be a dictionary with 'role' and 'content' keys.")


        explanation = query_groq_chat(prompt_messages)
        return jsonify({"explanation": explanation}) # jsonify automatically creates a Response

    except (ValueError, TypeError) as e:
        app.logger.warning(f"Bad request to /get_llm_explanation: {e}")
        return make_response(jsonify({"error": str(e)}), 400)
    except Exception as e: # Catches errors from query_groq_chat (e.g., Groq API issues)
        app.logger.error(f"Error in /get_llm_explanation: {e}", exc_info=True)
        # Distinguish upstream errors
        if "GROQ API" in str(e): # Check if the error message indicates an upstream Groq issue
            return make_response(jsonify({"error": f"Failed to get explanation from LLM. {str(e)}"}), 502) # Bad Gateway
        return make_response(jsonify({"error": "An internal server error occurred while fetching explanation."}), 500)


# In main.py
@app.route('/healthz', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200


# --- For local development/testing ONLY (remove for Render deployment) ---
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
