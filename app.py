import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
import ast
import torch

# --- Configuration ---
# Using pre-trained model for generating embeddings
# 'all-MiniLM-L6-v2' is a good balance of speed and performance for sentence embeddings
@st.cache_resource
def load_embedding_model():
    """Loads the SentenceTransformer model and caches it."""
    return SentenceTransformer('all-MiniLM-L6-v2')

# --- AST Parsing and Feature Extraction Helper ---

def extract_ast_features(tree):
    """
    Traverses AST and extracts relevant structural features.
    """
    features = {
        'has_try_except': False,
        'has_if': False,
        'has_for': False,
        'has_while': False,
        'function_names_called': [], # Simple list of function call names
        'exception_types_caught': [], # List of exception types in except clauses
        # Add more features as needed (e.g., variable assignments, imports)
    }

    if tree is None:
        return features # Return default features if parsing failed

    # Simple visitor pattern to find specific node types
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            # Capture the exception type if specified (e.g., except ValueError as e:)
            if node.type:
                 features['exception_types_caught'].append(ast.unparse(node.type).strip())
            features['has_try_except'] = True
        elif isinstance(node, ast.If):
            features['has_if'] = True
        elif isinstance(node, ast.For):
            features['has_for'] = True
        elif isinstance(node, ast.While):
            features['has_while'] = True
        elif isinstance(node, ast.Call):
             # Extract function name from a Call node
             if isinstance(node.func, ast.Name):
                 features['function_names_called'].append(node.func.id)
             elif isinstance(node.func, ast.Attribute):
                 # Handle method calls like obj.method()
                 features['function_names_called'].append(node.func.attr)


    # Remove duplicates from function_names_called
    features['function_names_called'] = list(set(features['function_names_called']))
    features['exception_types_caught'] = list(set(features['exception_types_caught']))

    return features


def parse_code_and_extract_ast_features(code_string):
    """
    Parses code to AST and extracts features.
    Uses st.warning for syntax errors.
    """
    tree = None
    features = {}
    try:
        tree = ast.parse(code_string)
        features = extract_ast_features(tree)
    except SyntaxError as e:
        st.warning(f"Could not parse code snippet into AST due to SyntaxError: {e}. Content: '{code_string[:100]}...'")
        # features will be empty/default if parsing fails
    return tree, features

# --- Data Simulation ---

# Sample Code Snippets (representing chunks from a repo)
code_snippets = [
    {"id": "code_1", "content": "def process_user_input(data):\n    # Validate input data\n    if not isinstance(data, dict):\n        raise ValueError('Input must be a dictionary')\n    user = data.get('user')\n    if not user:\n        raise ValueError('User data is missing')\n    return user", "metadata": {"file": "user_service.py", "function": "process_user_input"}, "ast": None, "ast_features": {}},
    {"id": "code_2", "content": "def save_user_to_db(user_data):\n    # Connect to database\n    db_connection = connect_db()\n    # Insert user data\n    db_connection.insert('users', user_data)\n    db_connection.close()", "metadata": {"file": "user_service.py", "function": "save_user_to_db"}, "ast": None, "ast_features": {}},
    {"id": "code_3", "content": "def connect_db():\n    # Simulate a database connection\n    print('Connecting to database...')\n    # This function might sometimes fail due to network issues\n    if random.random() < 0.1: # Simulate 10% failure rate\n        raise ConnectionError('Database connection failed')\n    return {'status': 'connected'}", "metadata": {"file": "database_utils.py", "function": "connect_db"}, "ast": None, "ast_features": {}},
    {"id": "code_4", "content": "def handle_request(request_data):\n    try:\n        user_info = process_user_input(request_data)\n        save_user_to_db(user_info)\n        return {'status': 'success'}\n    except Exception as e:\n        print(f'Error handling request: {e}')\n        # In a real system, this would log the error and stack trace\n        return {'status': 'error', 'message': str(e)}", "metadata": {"file": "api_handler.py", "function": "handle_request"}, "ast": None, "ast_features": {}},
    {"id": "code_5", "content": "def validate_config(config):\n    if not config.get('database_url'):\n        raise ValueError('Database URL missing in config')\n    # More validation logic...", "metadata": {"file": "config_loader.py", "function": "validate_config"}, "ast": None, "ast_features": {}},
     {"id": "code_6", "content": "def process_order_data(data):\n    # Similar validation logic for order data\n    if not isinstance(data, dict):\n        raise TypeError('Order data must be a dictionary')\n    order_id = data.get('order_id')\n    if not order_id:\n        raise ValueError('Order ID is missing')\n    return order_id", "metadata": {"file": "order_service.py", "function": "process_order_data"}, "ast": None, "ast_features": {}}, # Semantically similar to code_1 (validation)
]

# Simulated Stack Trace Frames (representing steps in execution)
simulated_stack_frames = [
    {"id": "st_frame_1", "content": "File \"api_handler.py\", line 10, in handle_request\n    user_info = process_user_input(request_data)", "metadata": {"file": "api_handler.py", "line": 10, "function": "handle_request"}},
    {"id": "st_frame_2", "content": "File \"user_service.py\", line 5, in process_user_input\n    if not isinstance(data, dict):", "metadata": {"file": "user_service.py", "line": 5, "function": "process_user_input"}},
    {"id": "st_frame_3", "content": "File \"user_service.py\", line 9, in process_user_input\n    if not user:", "metadata": {"file": "user_service.py", "line": 9, "function": "process_user_input"}},
    {"id": "st_frame_4", "content": "File \"api_handler.py\", line 11, in handle_request\n    save_user_to_db(user_info)", "metadata": {"file": "api_handler.py", "line": 11, "function": "handle_request"}},
    {"id": "st_frame_5", "content": "File \"user_service.py\", line 14, in save_user_to_db\n    db_connection = connect_db()", "metadata": {"file": "user_service.py", "line": 14, "function": "save_user_to_db"}},
    {"id": "st_frame_6", "content": "File \"database_utils.py\", line 6, in connect_db\n    raise ConnectionError('Database connection failed')", "metadata": {"file": "database_utils.py", "line": 6, "function": "connect_db"}},
    {"id": "st_frame_7", "content": "File \"order_service.py\", line 8, in process_order_data\n    if not isinstance(data, dict):", "metadata": {"file": "order_service.py", "line": 8, "function": "process_order_data"}}, # Semantically similar to st_frame_2 (input validation check)
]

# --- Embedding and Storage (Simulated Vector DB Setup) ---
@st.cache_data # Caches the result of this function
def prepare_and_embed_data(_raw_code_snippets, _raw_stack_frames, _embedding_model):
    """
    Parses AST for code snippets, embeds all content, and returns a simulated vector DB.
    """
    # 1. Parse AST for code snippets
    processed_code_snippets = []
    st.write("Parsing code snippets and extracting AST features...")
    for code_item_orig in _raw_code_snippets:
        code_item = code_item_orig.copy() # Avoid modifying original list of dicts
        tree, features = parse_code_and_extract_ast_features(code_item['content'])
        code_item['ast'] = tree # Store the tree (optional)
        code_item['ast_features'] = features # Store the extracted features
        processed_code_snippets.append(code_item)

    # 2. Embed and create vector_db
    db = []
    st.write("Populating vector database with embeddings...") # Optional progress

    def _add_to_db_internal(data_item, item_type, target_db, model_instance):
        content = data_item['content']
        embedding = model_instance.encode(content, convert_to_tensor=True)
        db_entry = {
            "id": data_item["id"],
            "type": item_type,
            "content": content,
            "metadata": data_item["metadata"],
            "embedding": embedding
        }
        if item_type == 'code' and 'ast_features' in data_item:
            db_entry['ast_features'] = data_item['ast_features']
        target_db.append(db_entry)

    for code_item in processed_code_snippets:
        _add_to_db_internal(code_item, 'code', db, _embedding_model)
    for frame_item in _raw_stack_frames:
        _add_to_db_internal(frame_item, 'stack_frame', db, _embedding_model)

    st.success(f"Vector database populated with {len(db)} items.")
    return db, processed_code_snippets

# --- Retrieval Logic ---

def retrieve_context_for_streamlit(
    error_stack_trace_content: list[str],
    error_message: str,
    k_val: int,
    current_vector_db: list,
    model_instance,
    similarity_threshold_val: float
):
    """
    Simulates retrieving relevant context from the vector DB based on an error stack trace.
    Retrieves relevant code chunks and semantically similar stack frames.
    Demonstrates using AST features to highlight relevant code.
    Returns structured data for Streamlit display.
    """
    st.info(f"Retrieving context for error: {error_message}") # Optional

    # 1. Embed the frames of the new error stack trace
    error_frame_embeddings = [model_instance.encode(frame_content, convert_to_tensor=True) for frame_content in error_stack_trace_content]

    # Prepare all embeddings from DB for efficient comparison
    db_embeddings = torch.stack([item['embedding'] for item in current_vector_db])

    retrieved_code_dict = {}
    retrieved_stack_frames_dict = {}

    # 2. For each frame in the error stack trace, find relevant context
    for i, frame_embedding in enumerate(error_frame_embeddings):
        st.write(f"Searching context for Frame {i+1}: {error_stack_trace_content[i].splitlines()[0]}...")

        # Search for similar items in the vector DB
        cosine_scores = util.cos_sim(frame_embedding.unsqueeze(0), db_embeddings)[0]

        # Get top-k results above the similarity threshold
        # Filter first by threshold, then sort and take top-k
        candidate_indices = [idx for idx, score in enumerate(cosine_scores) if score.item() > similarity_threshold_val]
        
        # Sort candidates by score and take top k_val
        sorted_candidates = sorted(candidate_indices, key=lambda idx: cosine_scores[idx].item(), reverse=True)
        top_results_indices = sorted_candidates[:k_val]

        if not top_results_indices:
            st.write("No relevant context found for this frame above the threshold.")
            continue

        st.write("Found relevant context for this frame:")
        for idx in top_results_indices:
            item = current_vector_db[idx]
            score = cosine_scores[idx].item()

            if item['type'] == 'code':
                if item['id'] not in retrieved_code_dict:
                    ast_features_info = ""
                    if "Error" in error_message or "Exception" in error_message:
                        if item.get('ast_features', {}).get('has_try_except'):
                            ast_features_info = " (AST: Has Try/Except)"
                        elif item.get('ast_features', {}).get('has_if'):
                            ast_features_info = " (AST: Has Conditional)"
                        elif item.get('ast_features', {}).get('has_for') or item.get('ast_features', {}).get('has_while'):
                            ast_features_info = " (AST: Has Loop)"
                    retrieved_code_dict[item['id']] = {"item": item, "score": score, "relevance_info": ast_features_info}

            elif item['type'] == 'stack_frame':
                if item['id'] not in retrieved_stack_frames_dict:
                    retrieved_stack_frames_dict[item['id']] = {"item": item, "score": score}

    return list(retrieved_code_dict.values()), list(retrieved_stack_frames_dict.values())

def run_streamlit_app():
    st.set_page_config(layout="wide", page_title="Error Context Retriever")
    st.title("Interactive Error Context Retrieval PoC")
    st.markdown("""
    This application demonstrates retrieving relevant code snippets and similar past stack traces
    based on a new error event. It uses sentence embeddings for semantic similarity and
    AST (Abstract Syntax Tree) features to identify structurally relevant code.
    """)

    # --- Load Model ---
    st.write("Loading embedding model...")
    embedding_model_instance = load_embedding_model()
    # Pass the actual model instance to prepare_and_embed_data
    vector_db_instance, _ = prepare_and_embed_data(code_snippets, simulated_stack_frames, embedding_model_instance)

    st.success("Embedding model loaded")
    st.sidebar.header("⚙️ Configuration")
    similarity_threshold_input = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.5, 0.01, help="Minimum cosine similarity score to consider an item relevant.")
    k_input = st.sidebar.number_input("Top-K Results per Frame", 1, 10, 3, help="Maximum number of relevant items to retrieve per stack frame.")

    st.sidebar.info(f"ℹ️ Vector DB contains **{len(vector_db_instance)}** items.")
    if st.sidebar.button("Show DB Item Counts"):
        code_count = sum(1 for item in vector_db_instance if item['type'] == 'code')
        frame_count = sum(1 for item in vector_db_instance if item['type'] == 'stack_frame')
        st.sidebar.write(f"- Code Snippets: {code_count}")
        st.sidebar.write(f"- Stack Frames: {frame_count}")

    st.header("🚨 Simulate New Error Event")

    error_examples = {
        "Select an example...": {"message": "", "trace": ""},
        "Database Connection Error": {
            "message": "Database connection failed",
            "trace": "File \"api_handler.py\", line 11, in handle_request\n    save_user_to_db(user_info)\nFile \"user_service.py\", line 14, in save_user_to_db\n    db_connection = connect_db()\nFile \"database_utils.py\", line 6, in connect_db\n    raise ConnectionError('Database connection failed')"
        },
        "User Input Validation Error": {
            "message": "Input must be a dictionary",
            "trace": "File \"api_handler.py\", line 10, in handle_request\n    user_info = process_user_input(request_data)\nFile \"user_service.py\", line 5, in process_user_input\n    if not isinstance(data, dict):"
        },
        "Order Input Validation Error": {
            "message": "Order data must be a dictionary",
            "trace": "File \"some_api.py\", line 25, in handle_order_request\n    order_details = process_order_data(request_payload)\nFile \"order_service.py\", line 8, in process_order_data\n    if not isinstance(data, dict):"
        }
    }
    selected_error_key = st.selectbox("Load Example Error (Optional):", list(error_examples.keys()))

    default_message = error_examples[selected_error_key]["message"]
    default_trace = error_examples[selected_error_key]["trace"]

    error_message_input = st.text_input("Error Message:", value=default_message, placeholder="e.g., NullPointerException")
    error_stack_trace_input = st.text_area("Error Stack Trace (one frame per line):", value=default_trace, height=150, placeholder="e.g., File \"example.py\", line 42, in my_function\n    problematic_call()")

    if st.button("🔍 Retrieve Context", type="primary"):
        if not error_message_input or not error_stack_trace_input:
            st.warning("⚠️ Please provide both an error message and a stack trace.")
        else:
            stack_trace_lines = [line.strip() for line in error_stack_trace_input.splitlines() if line.strip()]
            if not stack_trace_lines:
                st.warning("⚠️ Stack trace input is empty or contains only whitespace.")
            else:
                with st.spinner("🧠 Analyzing error and retrieving context..."):
                    retrieved_code_results, retrieved_frame_results = retrieve_context_for_streamlit(
                        stack_trace_lines,
                        error_message_input,
                        k_input,
                        vector_db_instance,
                        embedding_model_instance,
                        similarity_threshold_input
                    )

                st.header("💡 Retrieved Context")
                
                if not retrieved_code_results and not retrieved_frame_results:
                    st.info("No relevant context found based on the provided input and current settings.")

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("📝 Relevant Code Snippets")
                    if retrieved_code_results:
                        for res in sorted(retrieved_code_results, key=lambda x: x['score'], reverse=True): # Sort by score
                            item = res["item"]
                            st.markdown(f"**{item['metadata']['file']}::{item['metadata']['function']}**")
                            st.caption(f"Score: {res['score']:.4f}{res.get('relevance_info', '')}")
                            with st.expander("Show Code", expanded=False):
                                st.code(item['content'], language='python')
                    else:
                        st.info("No relevant code snippets found.")

                with col2:
                    st.subheader("👣 Similar Past Stack Frames")
                    if retrieved_frame_results:
                        for res in sorted(retrieved_frame_results, key=lambda x: x['score'], reverse=True): # Sort by score
                            item = res["item"]
                            st.markdown(f"**Frame from: {item['metadata'].get('file', 'N/A')}::{item['metadata'].get('function', 'N/A')}**")
                            st.caption(f"Score: {res['score']:.4f}")
                            with st.expander("Show Frame Content", expanded=False):
                                st.code(item['content'])
                    else:
                        st.info("No semantically similar stack frames found.")
    else:
        st.info("Enter error details above and click 'Retrieve Context' to begin.")

if __name__ == "__main__":
    try:
        import torch
        _ = torch.tensor([1.0])
    except ImportError:
        st.error("🚨 PyTorch is not installed or not working correctly. This application requires PyTorch. Please install it (`pip install torch torchvision torchaudio`) and try again.")
        st.stop()
    except Exception as e:
        st.error(f"🚨 Error initializing PyTorch: {e}. Please check your PyTorch installation.")
        st.stop()
        
    run_streamlit_app()
