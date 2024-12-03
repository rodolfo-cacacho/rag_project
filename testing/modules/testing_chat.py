def initialize_test_results(sql_con,test_name,test_results_table,test_results_table_schema,sql_eval_table):
    """
    Ensures the test_results table exists and inserts missing (test_name, question_id) pairs.
    """
    # Create the test_results table if it doesn't exist
    sql_con.create_table(test_results_table, test_results_table_schema)

    # Fetch all active questions
    questions = sql_con.get_records(sql_eval_table, ["id_question"], {"valid": 1})

    # Ensure questions are processed as dictionaries
    question_ids = [question["id_question"] if isinstance(question, dict) else question[0] for question in questions]

    # Fetch all existing records for this test
    existing_records = sql_con.get_records(
        test_results_table,
        ["id_question"],
        {"test_name": test_name}
    )
    existing_question_ids = {
        record["id_question"] if isinstance(record, dict) else record[0]
        for record in existing_records
    }

    # Find missing question IDs
    missing_question_ids = set(question_ids) - existing_question_ids

    if missing_question_ids:

        # Prepare records for batch insertion
        records_to_insert = [
            {
                "test_name": test_name,
                "id_question": question_id,
                "status": "pending"
            }
            for question_id in missing_question_ids
        ]

        # Insert all missing records at once
        if records_to_insert:
            sql_con.insert_many_records(test_results_table, records_to_insert)


def fetch_pending_questions(sql_con,test_name,test_table,qas_table):
    """
    Fetch all questions with status 'pending' for the given test.
    """
    # Retrieve pending questions using the get_records method
    conditions = {"test_name": test_name, "status": "pending"}
    records = sql_con.get_records(
        test_table,
        ["id_question"],
        conditions
    )

    if not records:
        return []

    # If records are dictionaries, access with keys
    if isinstance(records[0], dict):
        question_ids = [record["id_question"] for record in records]
    else:  # Otherwise, assume they are tuples and access by index
        question_ids = [record[0] for record in records]

    # Fetch question details for the extracted question IDs
    question_records = sql_con.get_records(
        qas_table,
        ["id_question", "question"],
        {"id_question": tuple(question_ids)}
    )

    return question_records

def update_question_status(sql_con,test_name, question_id, status, test_table,prompt_id,device,error_message=None):
    """
    Update the status and error_message of a question in test_results.
    """
    update_data = {"status": status,"prompt_id":prompt_id,"device":device}
    if error_message:
        update_data["error_message"] = error_message
    conditions = {"test_name": test_name, "id_question": question_id}
    sql_con.update_record(test_table, update_data, conditions)