import streamlit as st
import pandas as pd
import io
from datetime import datetime

def render_export_section(query_text):
    export_data = st.session_state.export_data

    # Individual test results
    if st.session_state.test_results:
        for i, result in enumerate(st.session_state.test_results):
            unique_id = f"Individual_{result.get('prompt_name', 'Unknown')}_{result['timestamp']}_{i}"
            if not any(d.get('unique_id') == unique_id for d in export_data):
                export_data.append({
                    'unique_id': unique_id,
                    'test_type': 'Individual',
                    'prompt_name': result.get('prompt_name', 'Unknown'),
                    'system_prompt': result['system_prompt'],
                    'query': result['query'],
                    'response': result['response'],
                    'status': result['status'],
                    'status_code': result.get('status_code', 'N/A'),
                    'timestamp': result['timestamp'],
                    'edited': result.get('edited', False),
                    'step': None,
                    'input_query': None,
                    'rating': st.session_state.response_ratings.get(f"test_{i}", 0) * 10,
                    'remark': result.get('remark', 'Saved and ran')
                })

    # Chain results
    if st.session_state.chain_results:
        for j, result in enumerate(st.session_state.chain_results):
            unique_id = f"Chain_{result.get('prompt_name', 'Unknown')}_{result['timestamp']}_{result.get('step')}"
            if not any(d.get('unique_id') == unique_id for d in export_data):
                export_data.append({
                    'unique_id': unique_id,
                    'test_type': 'Chain',
                    'prompt_name': result.get('prompt_name', 'Unknown'),
                    'system_prompt': result['system_prompt'],
                    'query': query_text,
                    'response': result['response'],
                    'status': result['status'],
                    'status_code': result.get('status_code', 'N/A'),
                    'timestamp': result['timestamp'],
                    'edited': result.get('edited', False),
                    'step': result.get('step'),
                    'input_query': result.get('input_query'),
                    'rating': st.session_state.response_ratings.get(f"chain_{j}", 0) * 10,
                    'remark': result.get('remark', 'Saved and ran')
                })

    # Combination results
    if st.session_state.combination_results:
        combination_data = st.session_state.combination_results
        
        if combination_data.get('individual_results'):
            for j, result in enumerate(combination_data['individual_results']):
                unique_id = f"Combination_Individual_{result.get('prompt_name', 'Unknown')}_{combination_data['timestamp']}_{j}"
                if not any(d.get('unique_id') == unique_id for d in export_data):
                    export_data.append({
                        'unique_id': unique_id,
                        'test_type': 'Combination_Individual',
                        'prompt_name': result.get('prompt_name', 'Unknown'),
                        'system_prompt': result.get('system_prompt', ''),
                        'query': query_text,
                        'response': result['response'],
                        'status': result['status'],
                        'status_code': result.get('status_code', 'N/A'),
                        'timestamp': combination_data['timestamp'],
                        'edited': result.get('edited', False),
                        'step': None,
                        'input_query': None,
                        'combination_strategy': combination_data.get('strategy'),
                        'combination_temperature': combination_data.get('temperature'),
                        'slider_weights': str(combination_data.get('slider_weights')) if combination_data.get('slider_weights') else None,
                        'rating': st.session_state.response_ratings.get(f"combination_individual_{j}", 0) * 10,
                        'remark': result.get('remark', 'Saved and ran')
                    })
        
        if combination_data.get('combined_result'):
            unique_id = f"Combination_Combined_{combination_data['timestamp']}"
            if not any(d.get('unique_id') == unique_id for d in export_data):
                export_data.append({
                    'unique_id': unique_id,
                    'test_type': 'Combination_Combined',
                    'prompt_name': 'AI_Combined',
                    'system_prompt': combination_data['combined_prompt'],
                    'query': query_text,
                    'response': combination_data['combined_result']['response'],
                    'status': combination_data['combined_result']['status'],
                    'status_code': combination_data['combined_result'].get('status_code', 'N/A'),
                    'timestamp': combination_data['timestamp'],
                    'edited': combination_data['combined_result'].get('edited', False),
                    'step': None,
                    'input_query': None,
                    'combination_strategy': combination_data.get('strategy'),
                    'combination_temperature': combination_data.get('temperature'),
                    'slider_weights': str(combination_data.get('slider_weights')) if combination_data.get('slider_weights') else None,
                    'rating': st.session_state.response_ratings.get("combination_combined", 0) * 10,
                    'remark': combination_data['combined_result'].get('remark', 'Saved and ran')
                })

    if export_data:
        st.header("💾 Export Results")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Download Excel", type="primary"):
                df = pd.DataFrame(export_data)
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Test_Results', index=False)
                    worksheet = writer.sheets['Test_Results']
                    for column in worksheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass
                        adjusted_width = min(max_length + 2, 50)
                        worksheet.column_dimensions[column_letter].width = adjusted_width
                excel_data = output.getvalue()
                st.download_button(
                    label="Download Excel File",
                    data=excel_data,
                    file_name=f"enhanced_api_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        with col2:
            if st.button("📋 Show DataFrame"):
                df = pd.DataFrame(export_data)
                st.dataframe(df, use_container_width=True)