import json
import argparse
import sys
import random

#from data_helpers import chunkenizer
max_answer_length = 5

def _clean_annotation(annotation):
    annotation["long_answer"] = _remove_html_byte_offsets(
        annotation["long_answer"])
    annotation["short_answers"] = [
        _remove_html_byte_offsets(sa) for sa in annotation["short_answers"]
    ]
    return annotation

def _remove_html_byte_offsets(span):
    if "start_byte" in span:
      del span["start_byte"]

    if "end_byte" in span:
      del span["end_byte"]

    return span

def process_nq_file(nq_filename, new_nq_filename, chunking, chunk_size, chunk_overlap, max_length, max_answer_length):
    '''
    This function processes the Natural Questions (NQ) dataset and saves the processed data to a new file with the structure:

    {
        "data": [
            {
                "question": "question text",
                "ctxs": ["document tokens"],
                "answers": ["short answer 1", "short answer 2", ...]            
            }
        ]
    }

    '''

    chunk_params = {'separators':['.'," ", ","], 'chunk_size':chunk_size, 'chunk_overlap':chunk_overlap}
     
    # Create new dictionary to save
    new_nq_dict = {'data': [], 'chunk_parameters':chunk_params}

    with open(nq_filename, 'r') as f:
        for line in f:
            data = json.loads(line)
            
            # Extract data from the json file
            question = data['question_text']
            doc_tokens = data['document_tokens']
            tokens_text = [t['token'] for t in data['document_tokens'] if not t["html_token"]]
            # Create a dict for this question
            question_dict = {}
            # Add question text
            question_dict['question'] = question
            question_dict['ctxs'] = tokens_text
            # Get short answer start and end tokens
            short_answers_info = []
            short_answer_texts = []
            if 'annotations' in data and len(data['annotations']) > 0:
                for annotation in data['annotations']:
                    if len(annotation['short_answers']) > 0:
  #                      short_answer_texts = []
                        print(annotation['short_answers'])  
                        annotation = _clean_annotation(annotation) 
                        for i in annotation['short_answers']:
                            s_start_token = i["start_token"]
                            s_end_token = i["end_token"]
                            s_answer_dicts = doc_tokens[s_start_token:s_end_token]
                            s_answer_tokens = [t['token'] for t in s_answer_dicts]
                            
                            if len(s_answer_tokens)<max_answer_length:
                                short_answer_texts.append(" ".join(s_answer_tokens))
                        
                question_dict['answers'] = short_answer_texts
                         

                # Check if the total number of tokens is less than 10,000
                if len(question_dict['ctxs']) < max_length:
                    if len(short_answer_texts)>0:
                    #short_answers_info.extend(annotation['short_answers'])
                        print(["[INFO]: Doc created and appended..."])
                        # Add this question dict to the list
                    
                        new_nq_dict['data'].append(question_dict)

            else:
                print(["[INFO]: Doc skipped due to length..."])

    # Shuffle the data
    random.shuffle(new_nq_dict['data'])

    # Split the data into dev and test sets
    dev_data = new_nq_dict['data'][:25]
    test_data = new_nq_dict['data'][25:125]

    # Save the dev data to a file
    print("[INFO]: Saving dev json file...")
    with open(f'{new_nq_filename}_dev_data.json', 'w') as f:
        json.dump(dev_data, f)
    print("[INFO]: dev json file saved...")

    # Save the test data to a file
    print("[INFO]: Saving test json file...")
    with open(f'{new_nq_filename}_test_data.json', 'w') as f:
        json.dump(test_data, f)
    print("[INFO]: test json file saved...")


    # # Save the new dictionary to a file
    # print(["[INFO]: Saving json file..."])
    # with open(new_nq_filename, 'w') as f:
    #     json.dump(new_nq_dict, f)
    # print(["[INFO]: json file saved..."])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process NQ file.')
    parser.add_argument('nq_filename', type=str, help='Path to the NQ file to process.')
    parser.add_argument('new_nq_filename', type=str, help='Path to save the processed NQ file.')
    parser.add_argument('chunking', type=bool, help='Whether to chunk the documents or not.')
    parser.add_argument('chunk_size', type=int, help='Chunk size')
    parser.add_argument('chunk_overlap', type=int,  help='Chunk overlap')
    parser.add_argument('max_length', type=int,  help='Maximum number of tokens to keep in a document.')
    parser.add_argument('max_answer_length', type=int,  help='Maximum number of answer tokens to keep in a document.')

    args = parser.parse_args()

    process_nq_file(args.nq_filename, args.new_nq_filename, args.chunking, args.chunk_size, args.chunk_overlap, args.max_length, args.max_answer_length)
    # Example use from commandline:
    # python process_nq.py /path/to/nq/file /path/to/new/nq/file True 100 50 10000
    # python '/Users/xmrt/Desktop/Master thesis/thesis-xmrt/utils/process_nq.py' 'data/nq-original-dev/v1.0-simplified-nq-dev-all.jsonl' data/nq-modified-dev/simplified-nq-dev-all-processed-10000.json' False 0 0 10000