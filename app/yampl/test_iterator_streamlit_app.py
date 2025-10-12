# Fixed AST-based static analysis with robust payload detection and smart YAML parser
import streamlit as st
import pandas as pd
import yaml
import json
import requests
import importlib.util
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
from io import BytesIO
import tempfile
import traceback
from typing import Optional, Dict, Any, List, Tuple
from pydantic import BaseModel, Field, ValidationError
import ast
import re
import logging
from itertools import product
from copy import deepcopy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Gemini
if os.getenv("GOOGLE_API_KEY"):
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
else:
    logger.warning("GOOGLE_API_KEY not set – LLM fallback will be disabled.")

# --- Pydantic Models ---
class PayloadParam(BaseModel):
    name: str = Field(..., description="Exact parameter key used in payload")
    type_hint: Optional[str] = Field(None, description="Type hint or expected type")
    required: bool = Field(False, description="Whether parameter is required")
    description: Optional[str] = None

class PayloadSpec(BaseModel):
    is_api: bool
    api_endpoint: Optional[str] = None
    http_method: Optional[str] = None
    function_name: Optional[str] = None
    execution_type: str
    required_params: List[str] = []
    system_prompt_param: Optional[str] = None
    query_param: Optional[str] = None
    additional_params: Dict[str, Any] = {}
    import_needed: Optional[str] = None

class AnalysisResult(BaseModel):
    raw: Dict[str, Any]
    spec: List[PayloadSpec]

# --- Smart YAML Parser Tool ---
class SmartYAMLParserTool:
    """
    Intelligently parses YAML files to extract system prompts and understand structure.
    Handles nested structures, lists, and automatically identifies prompt-like content.
    """
    
    def __init__(self):
        self.prompt_indicators = [
            'system_prompt', 'prompt', 'instruction', 'backstory', 'goal', 
            'system', 'personality', 'behavior', 'role', 'description'
        ]
    
    def parse_yaml_file(self, yaml_content: str) -> Dict[str, Any]:
        """Parse YAML and return structured data"""
        try:
            data = yaml.safe_load(yaml_content)
            return data
        except Exception as e:
            logger.error(f"YAML parsing error: {e}")
            return {}
    
    def identify_prompt_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze YAML structure to identify where prompts are stored.
        Returns a schema describing the structure.
        """
        schema = {
            'type': 'unknown',
            'param_mapping': {},
            'has_multiple_values': False,
            'iteration_strategy': None
        }
        
        if not isinstance(data, dict):
            return schema
        
        # Check for common patterns
        for key, value in data.items():
            if any(indicator in key.lower() for indicator in self.prompt_indicators):
                if isinstance(value, dict):
                    # Nested structure like system_prompts.general_agent.goal
                    schema['type'] = 'nested'
                    schema['param_mapping'] = self._analyze_nested_structure(value)
                    schema['iteration_strategy'] = 'nested'
                elif isinstance(value, list):
                    # List of prompts
                    schema['type'] = 'list'
                    schema['param_mapping'][key] = value
                    schema['has_multiple_values'] = True
                    schema['iteration_strategy'] = 'list'
                else:
                    # Simple string
                    schema['type'] = 'simple'
                    schema['param_mapping'][key] = value
        
        return schema
    
    def _analyze_nested_structure(self, data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Recursively analyze nested dictionary structure"""
        param_map = {}
        
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                # Recurse deeper
                nested = self._analyze_nested_structure(value, full_key)
                param_map.update(nested)
            elif isinstance(value, list):
                # Found a list - potential iteration point
                param_map[full_key] = {
                    'type': 'list',
                    'values': value,
                    'count': len(value)
                }
            else:
                # Leaf value
                param_map[full_key] = {
                    'type': 'value',
                    'value': value
                }
        
        return param_map
    
    def extract_all_prompt_variations(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract all possible prompt combinations from YAML.
        
        Examples:
        - If goals: [g1, g2, g3] and backstories: [b1, b2, b3]
          Returns: [{goal: g1, backstory: b1}, {goal: g2, backstory: b2}, {goal: g3, backstory: b3}]
        
        - If multiple lists exist, creates Cartesian product
        """
        schema = self.identify_prompt_structure(data)
        param_mapping = schema['param_mapping']
        
        logger.info(f"📋 YAML Schema detected: {schema['type']}")
        logger.info(f"📋 Param mapping: {json.dumps(param_mapping, indent=2, default=str)}")
        
        variations = []
        
        if schema['type'] == 'nested':
            # Handle nested structures
            variations = self._extract_from_nested(param_mapping)
        elif schema['type'] == 'list':
            # Handle simple lists
            for key, values in param_mapping.items():
                if isinstance(values, list):
                    for val in values:
                        variations.append({key: val})
        elif schema['type'] == 'simple':
            # Single value
            variations = [param_mapping]
        else:
            # Unknown structure - try to extract anything string-like
            variations = self._fallback_extraction(data)
        
        logger.info(f"✅ Extracted {len(variations)} prompt variation(s)")
        return variations
    
    def _extract_from_nested(self, param_map: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract variations from nested parameter mapping"""
        # Separate list params from value params
        list_params = {}
        value_params = {}
        
        for key, val in param_map.items():
            if isinstance(val, dict):
                if val.get('type') == 'list':
                    list_params[key] = val['values']
                else:
                    value_params[key] = val['value']
            else:
                value_params[key] = val
        
        if not list_params:
            # No lists, just return the values
            return [value_params] if value_params else []
        
        # Group parameters by their semantic meaning
        # E.g., goals and backstories should be in same group
        param_groups = self._group_related_params(list_params)
        
        logger.info(f"🔄 Parameter groups identified: {list(param_groups.keys())}")
        
        # Check if all lists in same group have same length (parallel iteration)
        # Otherwise use Cartesian product
        all_variations = []
        
        if len(param_groups) == 1:
            # Single group - check if parallel or cartesian
            group_name, group_params = list(param_groups.items())[0]
            list_lengths = [len(v) for v in group_params.values()]
            
            if len(set(list_lengths)) == 1 and list_lengths[0] > 1:
                # Parallel iteration - same length lists
                logger.info(f"🔄 Using PARALLEL iteration for group '{group_name}' (length: {list_lengths[0]})")
                for i in range(list_lengths[0]):
                    variant = deepcopy(value_params)
                    for key, values in group_params.items():
                        variant[key] = values[i]
                    all_variations.append(variant)
            else:
                # Cartesian product within group
                logger.info(f"🔄 Using CARTESIAN PRODUCT for group '{group_name}'")
                for combination in product(*group_params.values()):
                    variant = deepcopy(value_params)
                    for key, val in zip(group_params.keys(), combination):
                        variant[key] = val
                    all_variations.append(variant)
        else:
            # Multiple groups - create cartesian product across groups
            logger.info(f"🔄 Using CARTESIAN PRODUCT across {len(param_groups)} groups")
            
            # Get all combinations for each group
            group_combinations = {}
            for group_name, group_params in param_groups.items():
                group_combos = []
                for combination in product(*group_params.values()):
                    combo_dict = {}
                    for key, val in zip(group_params.keys(), combination):
                        combo_dict[key] = val
                    group_combos.append(combo_dict)
                group_combinations[group_name] = group_combos
            
            # Now create cartesian product of all group combinations
            all_group_combos = list(product(*group_combinations.values()))
            for combo_tuple in all_group_combos:
                variant = deepcopy(value_params)
                for combo_dict in combo_tuple:
                    variant.update(combo_dict)
                all_variations.append(variant)
        
        logger.info(f"✅ Created {len(all_variations)} total variations")
        return all_variations
    
    def _group_related_params(self, list_params: Dict[str, List[Any]]) -> Dict[str, Dict[str, List[Any]]]:
        """
        Group related parameters together.
        E.g., 'general_agent.goals' and 'general_agent.backstories' belong to same agent group.
        """
        groups = {}
        
        for key, values in list_params.items():
            # Extract the parent path (everything before the last dot)
            parts = key.split('.')
            
            if len(parts) > 1:
                # Has a parent, e.g., 'general_agent.goals' -> group: 'general_agent'
                group_name = '.'.join(parts[:-1])
                param_name = parts[-1]
            else:
                # No parent, use the key itself as both group and param
                group_name = 'default'
                param_name = key
            
            if group_name not in groups:
                groups[group_name] = {}
            
            groups[group_name][key] = values
        
        return groups
    
    def _fallback_extraction(self, data: Any) -> List[Dict[str, Any]]:
        """Fallback: extract any string values that look like prompts"""
        prompts = []
        
        def extract_strings(obj, path=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    new_path = f"{path}.{k}" if path else k
                    extract_strings(v, new_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    extract_strings(item, f"{path}[{i}]")
            elif isinstance(obj, str) and len(obj) > 20:  # Likely a prompt
                prompts.append({'prompt': obj, 'source': path})
        
        extract_strings(data)
        return prompts if prompts else [{'prompt': str(data)}]
    
    def map_to_code_params(self, variations: List[Dict[str, Any]], 
                          code_spec: PayloadSpec) -> List[Dict[str, Any]]:
        """
        Map extracted YAML variations to code parameters.
        
        Args:
            variations: List of extracted prompt variations
            code_spec: The analyzed code specification
        
        Returns:
            List of payloads ready to be used
        """
        mapped_payloads = []
        
        for variant in variations:
            payload = {}
            
            # Try to intelligently map YAML keys to code params
            for yaml_key, yaml_value in variant.items():
                yaml_key_lower = yaml_key.lower().split('.')[-1]  # Get last part of nested key
                
                # Map to system_prompt_param
                if code_spec.system_prompt_param:
                    if any(indicator in yaml_key_lower for indicator in 
                           ['goal', 'backstory', 'instruction', 'system', 'prompt', 'personality']):
                        if code_spec.system_prompt_param not in payload:
                            payload[code_spec.system_prompt_param] = yaml_value
                        else:
                            # Append if multiple prompt-like values
                            payload[code_spec.system_prompt_param] += f"\n{yaml_value}"
                
                # Store original mapping for reference
                payload[f"_yaml_{yaml_key}"] = yaml_value
            
            # Remove temp keys
            cleaned_payload = {k: v for k, v in payload.items() if not k.startswith('_yaml_')}
            
            # Store metadata
            mapped_payloads.append({
                'payload': cleaned_payload,
                'metadata': {k: v for k, v in payload.items() if k.startswith('_yaml_')},
                'raw_variant': variant
            })
        
        return mapped_payloads

# --- Utility functions ---
def safe_json_loads(s: str):
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r'\{.*\}', s, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None

def normalize_host(url: str) -> str:
    if not url:
        return url
    return url.replace("0.0.0.0", "localhost")

def extract_string(node: ast.AST) -> Optional[str]:
    """Return string value if node is ast.Constant or ast.Str."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    # For older Python versions
    if hasattr(ast, 'Str') and isinstance(node, ast.Str):
        return node.s
    return None

# --- Enhanced Smart Inspector with Robust Detection ---
class SmartInspectorTool:
    def __init__(self, llm_client=None):
        self.llm = llm_client
        self.route_decorators = ("app.post", "app.get", "app.put", "app.delete", 
                                "app.patch", "app.route", "router.post", "router.get", 
                                "router.api_route")
        # Variables to exclude (typically response/internal vars)
        self.exclude_vars = {
            'result', 'results', 'response', 'resp', 'res', 'candidates', 
            'distances', 'metadatas', 'documents', 'embedding', 'embeddings',
            'similar_docs', 'context', 'prompt', 'payload', 'headers',
            'generated_text', 'answer', 'content_text', 'url', 'client',
            'collection', 'model', 'tree', 'node', 'error', 'e', 'i', 'doc',
            'status_code', 'text', 'json', 'data', 'content', 'parts'
        }

    def parse_ast(self, code: str) -> Dict[str, Any]:
        """Enhanced AST parsing with better filtering."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            logger.error(f"Syntax error parsing code: {e}")
            return {}
            
        routes = []
        functions = []
        requests_calls = []
        model_classes = {}
        uvicorn_info = {}
        
        def get_decorator_name(dec):
            if isinstance(dec, ast.Call):
                func = dec.func
            else:
                func = dec
            if isinstance(func, ast.Attribute):
                v = []
                cur = func
                while isinstance(cur, ast.Attribute):
                    v.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name):
                    v.append(cur.id)
                return ".".join(reversed(v))
            if isinstance(func, ast.Name):
                return func.id
            return None

        # 1. Collect Pydantic BaseModel subclasses (REQUEST models only)
        logger.info("=" * 60)
        logger.info("STEP 1: Searching for Pydantic BaseModel classes...")
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                base_names = []
                for b in node.bases:
                    if isinstance(b, ast.Attribute):
                        names = []
                        cur = b
                        while isinstance(cur, ast.Attribute):
                            names.append(cur.attr)
                            cur = cur.value
                        if isinstance(cur, ast.Name):
                            names.append(cur.id)
                        base_names.append(".".join(reversed(names)))
                    elif isinstance(b, ast.Name):
                        base_names.append(b.id)
                
                logger.info(f"Found class: {node.name} with bases: {base_names}")
                
                # Check if it's a BaseModel
                if any("BaseModel" in s for s in base_names):
                    class_name = node.name
                    logger.info(f"  → This is a BaseModel subclass!")
                    
                    # Filter: only include Request/Input/Payload models, exclude Response models
                    is_request = re.search(r"(Request|Input|Payload|Body|Query)", class_name, re.I)
                    is_response = re.search(r"(Response|Output|Result)", class_name, re.I)
                    
                    logger.info(f"  → is_request={bool(is_request)}, is_response={bool(is_response)}")
                    
                    if is_request and not is_response:
                        fields = {}
                        for item in node.body:
                            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                                field_name = item.target.id
                                # Get type annotation
                                type_hint = None
                                if item.annotation:
                                    if isinstance(item.annotation, ast.Name):
                                        type_hint = item.annotation.id
                                    elif isinstance(item.annotation, ast.Subscript):
                                        # Optional[str], List[str], etc.
                                        if isinstance(item.annotation.value, ast.Name):
                                            type_hint = item.annotation.value.id
                                
                                # Check if required (no default value)
                                required = item.value is None
                                fields[field_name] = {
                                    'type': type_hint,
                                    'required': required
                                }
                        model_classes[class_name] = fields
                        logger.info(f"  ✅ STORED: {class_name} with fields: {list(fields.keys())}")
                    else:
                        logger.info(f"  ❌ SKIPPED: {class_name} (not a request model)")

        # 2. Find uvicorn.run host/port
        logger.info("\nSTEP 2: Searching for uvicorn.run configuration...")
        for node in ast.walk(tree):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                if isinstance(call.func, ast.Attribute):
                    if call.func.attr == "run" and isinstance(call.func.value, ast.Name) and \
                       call.func.value.id == "uvicorn":
                        logger.info("  Found uvicorn.run call!")
                        for kw in call.keywords:
                            if kw.arg == "host":
                                host_val = extract_string(kw.value)
                                if host_val:
                                    uvicorn_info['host'] = host_val
                                    logger.info(f"    host = {host_val}")
                            if kw.arg == "port":
                                if isinstance(kw.value, ast.Constant):
                                    uvicorn_info['port'] = int(kw.value.value)
                                    logger.info(f"    port = {kw.value.value}")
        
        if not uvicorn_info:
            logger.info("  No uvicorn.run found, using defaults")
            uvicorn_info = {'host': 'localhost', 'port': 8000}

        # 3. Find route definitions with their request models
        logger.info("\nSTEP 3: Searching for route decorators...")
        logger.info(f"Total nodes in tree.body: {len(tree.body)}")
        
        # Use tree.body instead of ast.walk to get top-level functions with decorators
        for idx, node in enumerate(tree.body):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_name = node.name
                decorators = node.decorator_list
                func_args = [a.arg for a in node.args.args]
                
                logger.info(f"Processing node {idx}: {type(node).__name__} - {func_name}")
                logger.info(f"  Arguments: {func_args}")
                logger.info(f"  Decorators: {len(decorators)}")
                
                if not decorators:
                    logger.info(f"  ⚠️ No decorators found, skipping")
                    continue
                
                # Find Pydantic model in function parameters
                param_model = None
                for arg in node.args.args:
                    if arg.annotation:
                        if isinstance(arg.annotation, ast.Name):
                            model_name = arg.annotation.id
                            if model_name in model_classes:
                                param_model = model_name
                                logger.info(f"  Found Pydantic param: {arg.arg}: {model_name}")
                                break
                
                for dec in decorators:
                    dec_name = get_decorator_name(dec)
                    if dec_name:
                        logger.info(f"  Decorator: @{dec_name}")
                        
                        # Check if this matches any route pattern
                        matched = False
                        for route_token in self.route_decorators:
                            dec_lower = dec_name.lower()
                            token_lower = route_token.lower()
                            
                            if token_lower in dec_lower or dec_lower.endswith(token_lower.split(".")[-1]):
                                matched = True
                                logger.info(f"    ✓ Matches route pattern: {route_token}")
                                
                                path = None
                                method = None
                                
                                if isinstance(dec, ast.Call):
                                    if dec.args:
                                        s = extract_string(dec.args[0])
                                        logger.info(f"      First arg: {repr(s)}")
                                        if s and isinstance(s, str) and s.startswith("/"):
                                            path = s
                                            logger.info(f"      ✓ Path extracted: {path}")
                                    
                                    # Check for router prefix
                                    if isinstance(dec.func, ast.Attribute) and isinstance(dec.func.value, ast.Name):
                                        router_name = dec.func.value.id
                                        for n in tree.body:
                                            if isinstance(n, ast.Assign):
                                                for t in n.targets:
                                                    if isinstance(t, ast.Name) and t.id == router_name:
                                                        if isinstance(n.value, ast.Call) and \
                                                           isinstance(n.value.func, ast.Name) and \
                                                           n.value.func.id == "APIRouter":
                                                            for kw in n.value.keywords:
                                                                if kw.arg == "prefix":
                                                                    prefix = extract_string(kw.value)
                                                                    if prefix and path:
                                                                        path = prefix.rstrip("/") + path
                                    
                                    if isinstance(dec.func, ast.Attribute):
                                        method = dec.func.attr.upper()
                                
                                if not method:
                                    try:
                                        method = dec_name.split(".")[-1].upper()
                                        if method not in ["POST", "GET", "PUT", "DELETE", "PATCH"]:
                                            method = "POST"
                                    except Exception:
                                        method = "POST"
                                
                                if path:
                                    routes.append({
                                        "path": path,
                                        "method": method,
                                        "function": func_name,
                                        "args": func_args,
                                        "decorator": dec_name,
                                        "body": node.body,
                                        "param_model": param_model
                                    })
                                    logger.info(f"      ✅ ROUTE STORED: {method} {path} -> {func_name}() (model: {param_model})")
                                
                                break
                        
                        if not matched:
                            logger.info(f"    ❌ Does not match any route pattern")

        # 4. Find requests.* calls
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                func_name = None
                if isinstance(func, ast.Attribute):
                    if isinstance(func.value, ast.Name):
                        func_name = f"{func.value.id}.{func.attr}"
                
                if func_name and func_name.startswith("requests."):
                    url = None
                    http_method = func_name.split(".", 1)[1].upper()
                    if node.args:
                        url = extract_string(node.args[0])
                    
                    json_keys = []
                    for kw in node.keywords:
                        if kw.arg in ("json", "data"):
                            if isinstance(kw.value, ast.Dict):
                                for k in kw.value.keys:
                                    key_str = extract_string(k)
                                    if key_str and key_str not in self.exclude_vars:
                                        json_keys.append(key_str)
                    
                    requests_calls.append({
                        "method": http_method,
                        "url": url,
                        "json_keys": json_keys
                    })

        # 5. Extract REQUEST body keys
        body_keys = set()
        for route_info in routes:
            route_body = route_info.get('body', [])
            for node in ast.walk(ast.Module(body=route_body)):
                if isinstance(node, ast.Attribute):
                    if isinstance(node.value, ast.Name):
                        if node.value.id == 'request':
                            attr_name = node.attr
                            if attr_name not in self.exclude_vars:
                                body_keys.add(attr_name)
                
                if isinstance(node, ast.Subscript):
                    try:
                        index = node.slice
                        if isinstance(index, ast.Constant):
                            key = index.value
                        elif isinstance(index, ast.Index):
                            key = extract_string(index.value)
                        else:
                            key = extract_string(index)
                        
                        if isinstance(key, str) and key not in self.exclude_vars:
                            if isinstance(node.value, ast.Name):
                                if node.value.id in ('request', 'body', 'payload', 'data'):
                                    body_keys.add(key)
                    except Exception:
                        pass
                
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if node.func.attr == "get" and node.args:
                        key = extract_string(node.args[0])
                        if key and key not in self.exclude_vars:
                            if isinstance(node.func.value, ast.Name):
                                base = node.func.value.id
                                if base in ("body", "payload", "data", "request"):
                                    body_keys.add(key)

        logger.info("\n" + "=" * 60)
        logger.info(f"SUMMARY: Found {len(model_classes)} Pydantic models, {len(routes)} routes")
        logger.info(f"Models: {list(model_classes.keys())}")
        logger.info(f"Routes: {[(r['method'], r['path'], r['param_model']) for r in routes]}")
        logger.info("=" * 60 + "\n")
        
        return {
            "routes": routes,
            "functions": functions,
            "requests_calls": requests_calls,
            "model_classes": model_classes,
            "uvicorn": uvicorn_info,
            "body_keys": list(body_keys)
        }

    def synthesize_spec_from_ast(self, ast_info: Dict[str, Any]) -> Tuple[List[PayloadSpec], float]:
        """Build PayloadSpec list from AST with confidence score. Returns ALL matching routes."""
        routes = ast_info.get("routes", [])
        requests_calls = ast_info.get("requests_calls", [])
        model_classes = ast_info.get("model_classes", {})
        body_keys = ast_info.get("body_keys", [])
        uvicorn = ast_info.get("uvicorn", {})

        logger.info("\n" + "=" * 60)
        logger.info("SYNTHESIZING PAYLOAD SPEC")
        logger.info(f"Routes found: {len(routes)}")
        logger.info(f"Model classes: {list(model_classes.keys())}")
        logger.info(f"Uvicorn config: {uvicorn}")

        specs = []
        
        # Priority 1: FastAPI routes with Pydantic models
        if routes:
            logger.info("\n✓ Using FastAPI route detection")
            
            for route in routes:
                path = route.get("path")
                method = route.get("method") or "POST"
                func = route.get("function")
                param_model = route.get("param_model")
                
                logger.info(f"\nProcessing route: {method} {path}")
                
                required_params = []
                if param_model and param_model in model_classes:
                    model_fields = model_classes[param_model]
                    required_params = list(model_fields.keys())
                    logger.info(f"  ✓ Using Pydantic model '{param_model}' fields: {required_params}")
                else:
                    required_params = body_keys if body_keys else []
                    logger.info(f"  ⚠️ No Pydantic model, using body_keys: {required_params}")

                host = uvicorn.get("host", "localhost")
                port = uvicorn.get("port", 8000)
                api_endpoint = f"http://{host}:{port}{path}" if path else None
                
                logger.info(f"  API endpoint: {api_endpoint}")

                query_param = None
                system_prompt_param = None
                uses_prompt_key = False
                
                for k in required_params:
                    lk = k.lower()
                    if lk in ("query", "question", "input", "text", "message") and not query_param:
                        query_param = k
                        logger.info(f"    → Detected QUERY param: {k}")
                    if lk in ("system_prompt", "instruction", "prompt", "goal", "backstory") and not system_prompt_param:
                        system_prompt_param = k
                        logger.info(f"    → Detected SYSTEM_PROMPT param: {k}")
                    if lk == "system_prompt_key" or lk.endswith("_key"):
                        uses_prompt_key = True
                        logger.info(f"    → Detected PROMPT_KEY param: {k}")
                        
                spec = PayloadSpec(
                    is_api=True,
                    api_endpoint=normalize_host(api_endpoint) if api_endpoint else None,
                    http_method=method,
                    function_name=func,
                    execution_type="api",
                    required_params=required_params,
                    system_prompt_param=system_prompt_param,
                    query_param=query_param,
                    additional_params={"uses_prompt_key": uses_prompt_key, "route_path": path},
                )
                specs.append(spec)
                logger.info(f"  ✅ Spec created for {method} {path}")
            
            confidence = 0.95 if any(s.additional_params.get("uses_prompt_key") for s in specs) else 0.90
            logger.info(f"\n✅ Created {len(specs)} specs with confidence: {confidence}")
            logger.info("=" * 60 + "\n")
            
            return specs, confidence

        # Priority 2: requests.* calls
        if requests_calls:
            logger.info("\n✓ Using requests.* detection")
            rc = requests_calls[0]
            url = rc.get("url")
            method = rc.get("method", "POST")
            json_keys = rc.get("json_keys", [])
            
            api_endpoint = normalize_host(url) if url else None
            query_param = None
            system_prompt_param = None
            
            for k in json_keys:
                lk = k.lower()
                if lk in ("query", "message", "text", "input", "question"):
                    query_param = k
                if lk in ("system_prompt", "prompt", "instruction", "goal", "backstory"):
                    system_prompt_param = k
            
            spec = PayloadSpec(
                is_api=True,
                api_endpoint=api_endpoint,
                http_method=method,
                function_name=None,
                execution_type="api",
                required_params=json_keys,
                system_prompt_param=system_prompt_param,
                query_param=query_param,
                additional_params={}
            )
            return [spec], 0.80

        # Priority 3: Function-based
        functions = ast_info.get("functions", [])
        if functions:
            logger.info("\n✓ Using function detection")
            chosen = None
            for f in functions:
                if re.search(r"(rag|chat|run|handle|generate)", f.get("name", ""), re.I):
                    chosen = f
                    break
            
            if not chosen:
                chosen = functions[0]
            
            fname = chosen.get("name")
            args = chosen.get("args", [])
            
            query_param = None
            system_prompt_param = None
            for a in args:
                la = a.lower()
                if la in ("query", "message", "text", "input", "question"):
                    query_param = a
                if la in ("system_prompt", "system", "prompt", "goal", "backstory"):
                    system_prompt_param = a
            
            spec = PayloadSpec(
                is_api=False,
                api_endpoint=None,
                http_method=None,
                function_name=fname,
                execution_type="function",
                required_params=args,
                system_prompt_param=system_prompt_param,
                query_param=query_param,
                additional_params={}
            )
            return [spec], 0.70

        # Priority 4: Check if code is a script (has main execution logic)
        # Look for crew.kickoff(), agent initialization, or similar patterns
        logger.info("\n✓ Checking for script-based execution")
        has_crew = any("crew" in str(node).lower() for node in ast.walk(tree))
        has_agent = any("agent" in str(node).lower() for node in ast.walk(tree))
        
        if has_crew or has_agent:
            logger.info("  Detected agent/crew-based script")
            # Return a special spec for script execution
            spec = PayloadSpec(
                is_api=False,
                api_endpoint=None,
                http_method=None,
                function_name=None,  # Will execute as script
                execution_type="script",
                required_params=["query", "goal", "backstory"],
                system_prompt_param="goal",  # or backstory
                query_param="query",
                additional_params={"is_script": True}
            )
            return [spec], 0.60

        return [], 0.0

    def call_llm_for_clarification(self, code: str, hints: Dict[str, Any]) -> Optional[PayloadSpec]:
        """LLM fallback when static analysis fails."""
        if not self.llm:
            return None
        
        hint_text = json.dumps(hints, indent=2)
        prompt = (
            "Analyze this Python code and return a JSON describing the API endpoint or function.\n\n"
            "HINTS from AST:\n"
            f"{hint_text}\n\n"
            "CODE:\n"
            "'''\n"
            f"{code[:15000]}\n"
            "'''\n\n"
            "Return ONLY JSON:\n"
            "{\n"
            '  "is_api": true/false,\n'
            '  "api_endpoint": "http://localhost:8000/route" or null,\n'
            '  "http_method": "POST/GET" or null,\n'
            '  "function_name": "name" or null,\n'
            '  "required_params": ["param1", "param2"],\n'
            '  "system_prompt_param": "param_name" or null,\n'
            '  "query_param": "param_name" or null,\n'
            '  "execution_type": "api" or "function"\n'
            "}\n"
        )
        
        resp = self.llm.generate_text(prompt, max_output_tokens=800)
        parsed = safe_json_loads(resp)
        if not parsed:
            return None
        
        if parsed.get("api_endpoint"):
            parsed["api_endpoint"] = normalize_host(parsed["api_endpoint"])
        
        try:
            spec = PayloadSpec(
                is_api=bool(parsed.get("is_api")),
                api_endpoint=parsed.get("api_endpoint"),
                http_method=parsed.get("http_method"),
                function_name=parsed.get("function_name"),
                execution_type=parsed.get("execution_type", "unknown"),
                required_params=parsed.get("required_params", []) or [],
                system_prompt_param=parsed.get("system_prompt_param"),
                query_param=parsed.get("query_param"),
                additional_params=parsed.get("additional_params", {}) or {}
            )
            return spec
        except ValidationError as ve:
            logger.error(f"Pydantic validation error: {ve}")
            return None

    def inspect(self, code_content: str) -> AnalysisResult:
        """Main inspection with AST + LLM fallback. Returns ALL detected routes."""
        try:
            ast_info = self.parse_ast(code_content)
        except Exception as e:
            logger.exception("AST parse failed")
            ast_info = {}

        specs, conf = ([], 0.0)
        try:
            specs, conf = self.synthesize_spec_from_ast(ast_info)
            logger.info(f"AST analysis confidence: {conf}, found {len(specs)} specs")
        except Exception as e:
            logger.exception("synthesize_spec_from_ast error")
            specs, conf = ([], 0.0)

        if specs and conf > 0.5:
            return AnalysisResult(raw={"source": "ast", "confidence": conf, "hints": ast_info}, spec=specs)

        # LLM fallback
        if self.llm:
            try:
                logger.info("Attempting LLM fallback")
                spec_llm = self.call_llm_for_clarification(code_content, ast_info)
                if spec_llm:
                    return AnalysisResult(raw={"source": "llm", "hints": ast_info}, spec=[spec_llm])
            except Exception as e:
                logger.exception("LLM fallback failed")

        # Unknown fallback
        unknown_spec = PayloadSpec(
            is_api=False,
            api_endpoint=None,
            http_method=None,
            function_name=None,
            execution_type="unknown",
            required_params=[],
            system_prompt_param=None,
            query_param=None,
            additional_params={}
        )
        return AnalysisResult(raw={"source": "none", "hints": ast_info}, spec=[unknown_spec])

# --- LLM Client ---
class LLMClient:
    def __init__(self, model_name: str = "gemini-2.0-flash-exp"):
        self.model_name = model_name

    def generate_text(self, prompt: str, max_output_tokens: int = 1024) -> str:
        try:
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(prompt)
            if hasattr(response, "text") and response.text:
                return response.text
            return str(response)
        except Exception as e:
            logger.exception("LLM call failed")
            return f"LLM_ERROR: {e}"

# --- Payload Builder & Executor ---
class PayloadBuilderTool:
    def build(self, spec: PayloadSpec, system_prompt: str, query: str, 
              yaml_variant: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Build payload with optional YAML variant data.
        
        Args:
            spec: PayloadSpec from code analysis
            system_prompt: Default system prompt (used if no YAML variant)
            query: User query
            yaml_variant: Optional dict containing YAML-extracted prompt variations
        """
        payload = {}
        
        # If we have YAML variant data, use it to populate prompt fields
        if yaml_variant:
            logger.info(f"🎯 Building payload with YAML variant: {yaml_variant}")
            
            # Try to map YAML keys to spec params
            for yaml_key, yaml_value in yaml_variant.items():
                yaml_key_lower = yaml_key.lower().split('.')[-1]
                
                # Map goal/backstory/instruction to system_prompt_param
                if spec.system_prompt_param:
                    if any(indicator in yaml_key_lower for indicator in 
                           ['goal', 'backstory', 'instruction', 'system', 'prompt', 'personality', 'role']):
                        if spec.system_prompt_param not in payload:
                            payload[spec.system_prompt_param] = yaml_value
                        else:
                            # Combine multiple prompt-like fields
                            payload[spec.system_prompt_param] += f"\n\n{yaml_value}"
        else:
            # Use default system prompt
            if spec.system_prompt_param:
                payload[spec.system_prompt_param] = system_prompt
        
        # Always add query
        if spec.query_param:
            payload[spec.query_param] = query
        
        # Handle system_prompt_key case
        if spec.additional_params.get("uses_prompt_key"):
            logger.info(f"Uses prompt key - sending system prompt directly")
            # Don't include system_prompt_key in payload
            for p in spec.required_params:
                if p.lower() in ("system_prompt_key", "prompt_key") or p.lower().endswith("_key"):
                    continue
                if p not in payload:
                    lp = p.lower()
                    if lp in ("top_k", "topk", "k"):
                        payload[p] = 5
                    elif lp in ("temperature", "temp"):
                        payload[p] = 0.7
                    elif lp in ("max_tokens", "max_output_tokens"):
                        payload[p] = 150
                    else:
                        payload[p] = ""
            return payload
        
        # Fill other required params
        for p in spec.required_params:
            if p not in payload:
                lp = p.lower()
                if lp in ("top_k", "topk", "k"):
                    payload[p] = 5
                elif lp in ("temperature", "temp"):
                    payload[p] = 0.7
                elif lp in ("max_tokens", "max_output_tokens"):
                    payload[p] = 150
                else:
                    payload[p] = ""
        
        # Fallback if no explicit query param
        if not spec.query_param:
            for fallback in ("query", "message", "text", "input"):
                if fallback not in payload:
                    payload[fallback] = query
                    break
        
        return payload

class ExecutorTool:
    def execute(self, spec: PayloadSpec, payload: Dict[str, Any], 
                code_content: Optional[str] = None, timeout: int = 30) -> Dict[str, Any]:
        try:
            if spec.execution_type == "api" and spec.is_api:
                url = spec.api_endpoint
                method = (spec.http_method or "POST").upper()
                
                if not url:
                    return {"success": False, "error": "No API endpoint provided"}
                
                logger.info(f"Executing {method} {url} with payload: {payload}")
                
                if method == "POST":
                    r = requests.post(url, json=payload, timeout=timeout)
                else:
                    r = requests.get(url, params=payload, timeout=timeout)
                
                r.raise_for_status()
                
                try:
                    result = r.json()
                except Exception:
                    result = {"text": r.text}
                
                return {"success": True, "result": result, "status_code": r.status_code}
            
            elif spec.execution_type == "function" and spec.function_name and code_content:
                # Add safety check for problematic imports
                if 'crewai' in code_content.lower() or 'crew' in code_content.lower():
                    return {
                        "success": False, 
                        "error": "CrewAI detected - cannot execute due to dependency conflicts. Please test via API endpoint instead."
                    }
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(code_content)
                    temp_file = f.name
                
                try:
                    spec_module = importlib.util.spec_from_file_location("user_module", temp_file)
                    module = importlib.util.module_from_spec(spec_module)
                    sys.modules["user_module"] = module
                    spec_module.loader.exec_module(module)
                    
                    func = getattr(module, spec.function_name, None)
                    if not func:
                        return {"success": False, "error": f"Function {spec.function_name} not found"}
                    
                    result = func(**payload)
                    return {"success": True, "result": result}
                finally:
                    try:
                        os.unlink(temp_file)
                    except Exception:
                        pass
            else:
                return {"success": False, "error": "Unsupported execution type"}
        
        except Exception as e:
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

# --- Main Agent ---
class PromptTestingAgent:
    def __init__(self, llm_model: str = "gemini-2.0-flash-exp"):
        self.llm_client = LLMClient(model_name=llm_model) if os.getenv("GOOGLE_API_KEY") else None
        self.inspector = SmartInspectorTool(self.llm_client)
        self.builder = PayloadBuilderTool()
        self.executor = ExecutorTool()
        self.yaml_parser = SmartYAMLParserTool()

    def analyze_code(self, code_content: str) -> AnalysisResult:
        return self.inspector.inspect(code_content)

    def parse_prompts_yaml(self, yaml_content: str) -> List[Dict[str, Any]]:
        """Parse YAML and extract all prompt variations"""
        data = self.yaml_parser.parse_yaml_file(yaml_content)
        variations = self.yaml_parser.extract_all_prompt_variations(data)
        return variations

    def create_payload(self, spec: PayloadSpec, system_prompt: str, query: str,
                      yaml_variant: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.builder.build(spec, system_prompt, query, yaml_variant)

    def execute(self, spec: PayloadSpec, payload: Dict[str, Any], 
                code_content: Optional[str] = None) -> Dict[str, Any]:
        return self.executor.execute(spec, payload, code_content=code_content)

# --- File Loading ---
def load_from_file(uploaded_file) -> list:
    """Load content from various file formats."""
    try:
        file_extension = Path(uploaded_file.name).suffix.lower()

        if file_extension in ['.yaml', '.yml']:
            content = yaml.safe_load(uploaded_file)
            return flatten_yaml(content)

        elif file_extension == '.txt':
            content = uploaded_file.read().decode('utf-8')
            return [line.strip() for line in content.split('\n') if line.strip()]

        elif file_extension == '.csv':
            df = pd.read_csv(uploaded_file)
            return df.iloc[:, 0].dropna().astype(str).tolist()

        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(uploaded_file)
            return df.iloc[:, 0].dropna().astype(str).tolist()

        else:
            st.error(f"Unsupported file format: {file_extension}")
            return []

    except Exception as e:
        st.error(f"Error loading file: {e}")
        return []

def flatten_yaml(obj):
    """Recursively flatten YAML structure to list of values."""
    values = []
    if isinstance(obj, dict):
        for v in obj.values():
            values.extend(flatten_yaml(v))
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                values.extend(flatten_yaml(item))
            else:
                values.append(str(item))
    else:
        values.append(str(obj))
    return values

# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="Robust Prompt Testing Agent", layout="wide", page_icon="🧠")
    st.title("🧠 Prompt Testing Agent – Enhanced AST Analysis with Smart YAML Parser")
    st.markdown("Robust detection of FastAPI Pydantic models and intelligent YAML prompt iteration")

    # Session state
    if 'test_results' not in st.session_state:
        st.session_state.test_results = None
    if 'analysis' not in st.session_state:
        st.session_state.analysis = None
    if 'system_prompts' not in st.session_state:
        st.session_state.system_prompts = []
    if 'queries' not in st.session_state:
        st.session_state.queries = []
    if 'code_analyzed' not in st.session_state:
        st.session_state.code_analyzed = False
    if 'yaml_variations' not in st.session_state:
        st.session_state.yaml_variations = []
    if 'use_smart_yaml' not in st.session_state:
        st.session_state.use_smart_yaml = False

    # Sidebar
    with st.sidebar:
        st.header("📁 Upload Files")
        code_file = st.file_uploader("Upload RAG/Agent Code", type=['py'], 
                                     help="Upload your Python code (FastAPI or function-based)")
        st.divider()
        if os.getenv("GOOGLE_API_KEY"):
            st.success("✅ Gemini API Key loaded")
        else:
            st.info("LLM fallback disabled – AST-only analysis")

    # Code Analysis
    st.subheader("🔍 Code Analysis")
    col1, col2 = st.columns([1,1])

    with col1:
        if code_file:
            code_content = code_file.read().decode('utf-8')
            with st.expander("View Code", expanded=False):
                st.code(code_content, language='python')
            
            current_file_id = f"{code_file.name}_{code_file.size}"
            if not st.session_state.code_analyzed or \
               st.session_state.get('last_code_file_id') != current_file_id:
                with st.spinner("🔎 Analyzing code..."):
                    agent = PromptTestingAgent()
                    analysis = agent.analyze_code(code_content)
                    st.session_state.analysis = analysis
                    st.session_state.code_content = code_content
                    st.session_state.code_analyzed = True
                    st.session_state.last_code_file_id = current_file_id
        else:
            st.info("👆 Upload your code to begin")
            st.session_state.code_analyzed = False

    with col2:
        if st.session_state.analysis:
            st.markdown("**Analysis Results**")
            specs = st.session_state.analysis.spec
            
            if specs:
                st.success(f"**✅ Found {len(specs)} Endpoint(s)**")
                
                api_specs = [s for s in specs if s.is_api]
                func_specs = [s for s in specs if not s.is_api]
                
                if api_specs:
                    st.markdown("### 🌐 API Endpoints")
                    for idx, spec in enumerate(api_specs):
                        route_path = spec.additional_params.get('route_path', spec.api_endpoint)
                        st.info(f"**{spec.http_method}** `{route_path}`")
                        
                        if spec.required_params:
                            st.caption(f"Parameters: {', '.join(spec.required_params)}")
                    
                    with st.expander("📋 API Route Details", expanded=False):
                        for idx, spec in enumerate(api_specs):
                            route_path = spec.additional_params.get('route_path')
                            st.markdown(f"### {spec.http_method} {route_path}")
                            st.json(json.loads(spec.model_dump_json()))
                            
                            if spec.query_param:
                                st.info(f"**Query Param:** `{spec.query_param}`")
                            if spec.system_prompt_param:
                                st.info(f"**System Prompt Param:** `{spec.system_prompt_param}`")
                            
                            st.divider()
                
                if func_specs:
                    st.markdown("### 🔧 Functions")
                    for spec in func_specs:
                        st.info(f"**Function:** `{spec.function_name}()`")
                        if spec.required_params:
                            st.caption(f"Parameters: {', '.join(spec.required_params)}")
            else:
                st.error("⚠️ No endpoints or functions detected")

    st.divider()

    # System Prompts with Smart YAML Parser
    st.subheader("💬 System Prompts")
    
    # Toggle for smart YAML parsing
    use_smart_yaml = st.checkbox(
        "🧠 Use Smart YAML Parser (Auto-detect goals, backstories, etc.)",
        value=st.session_state.use_smart_yaml,
        help="Enable this to automatically parse and iterate through YAML structures with multiple goals/backstories"
    )
    st.session_state.use_smart_yaml = use_smart_yaml
    
    if use_smart_yaml:
        st.info("📚 **Smart YAML Mode**: Upload a YAML file with nested prompts. The agent will automatically detect and iterate through all variations.")
    
    tab1, tab2 = st.tabs(["📤 Upload File", "✏️ Manual Input"])
    
    with tab1:
        prompts_file = st.file_uploader("Upload System Prompts", 
                                       type=['yaml','yml','txt','csv','xlsx','xls'], 
                                       key="prompts_file")
        if prompts_file:
            current_file_id = f"{prompts_file.name}_{prompts_file.size}"
            if st.session_state.get('last_prompts_file_id') != current_file_id:
                with st.spinner("Loading prompts..."):
                    if use_smart_yaml and prompts_file.name.endswith(('.yaml', '.yml')):
                        # Use smart YAML parser
                        yaml_content = prompts_file.read().decode('utf-8')
                        agent = PromptTestingAgent()
                        variations = agent.parse_prompts_yaml(yaml_content)
                        st.session_state.yaml_variations = variations
                        st.session_state.system_prompts = []  # Clear old prompts
                        st.session_state.last_prompts_file_id = current_file_id
                        
                        st.success(f"✅ Parsed {len(variations)} prompt variation(s) from YAML")
                        
                        # Show parsed variations
                        with st.expander("📋 View Parsed Variations", expanded=True):
                            for i, var in enumerate(variations):
                                st.markdown(f"**Variation {i+1}:**")
                                st.json(var)
                        
                        st.rerun()
                    else:
                        # Use traditional loading
                        loaded_prompts = load_from_file(prompts_file)
                        if loaded_prompts:
                            st.session_state.system_prompts = loaded_prompts
                            st.session_state.yaml_variations = []
                            st.session_state.last_prompts_file_id = current_file_id
                            st.success(f"✅ Loaded {len(loaded_prompts)} prompts")
                            st.rerun()

    with tab2:
        manual_prompts = st.text_area("Enter System Prompts (one per line)", 
                                     height=200, 
                                     placeholder="You are a helpful assistant.\nYou are an expert.",
                                     key="manual_prompts")
        col5, col6 = st.columns([1,1])
        with col5:
            if st.button("➕ Add Manual Prompts", type="primary"):
                if manual_prompts.strip():
                    new_prompts = [line.strip() for line in manual_prompts.split('\n') if line.strip()]
                    st.session_state.system_prompts.extend(new_prompts)
                    st.session_state.yaml_variations = []  # Clear YAML variations
                    st.success(f"✅ Added {len(new_prompts)} prompts")
        with col6:
            if st.button("🗑️ Clear All Prompts"):
                st.session_state.system_prompts = []
                st.session_state.yaml_variations = []
                st.session_state.pop('last_prompts_file_id', None)
                st.success("✅ Cleared")
                st.rerun()

    # Display current prompt status
    if st.session_state.yaml_variations:
        st.success(f"**📚 Smart YAML Mode: {len(st.session_state.yaml_variations)} variations loaded**")
        st.caption("The agent will test each variation against all queries")
    elif st.session_state.system_prompts:
        st.success(f"**Loaded: {len(st.session_state.system_prompts)} prompts**")
        with st.expander("Edit Prompts", expanded=False):
            prompts_to_delete = []
            for i, prompt in enumerate(st.session_state.system_prompts):
                col_a, col_b = st.columns([5,1])
                with col_a:
                    edited = st.text_area(f"Prompt {i+1}", prompt, height=80, key=f"edit_prompt_{i}")
                    if edited != prompt:
                        st.session_state.system_prompts[i] = edited
                with col_b:
                    if st.button("🗑️", key=f"del_prompt_{i}"):
                        prompts_to_delete.append(i)
            for idx in sorted(prompts_to_delete, reverse=True):
                st.session_state.system_prompts.pop(idx)
            if prompts_to_delete:
                st.rerun()
    else:
        st.warning("⚠️ No prompts loaded")

    st.divider()

    # Queries
    st.subheader("❓ Test Queries")
    tab3, tab4 = st.tabs(["📤 Upload File", "✏️ Manual Input"])
    
    with tab3:
        queries_file = st.file_uploader("Upload Queries", 
                                       type=['yaml','yml','txt','csv','xlsx','xls'],
                                       key="queries_file")
        if queries_file:
            current_file_id = f"{queries_file.name}_{queries_file.size}"
            if st.session_state.get('last_queries_file_id') != current_file_id:
                with st.spinner("Loading queries..."):
                    loaded_queries = load_from_file(queries_file)
                    if loaded_queries:
                        st.session_state.queries = loaded_queries
                        st.session_state.last_queries_file_id = current_file_id
                        st.success(f"✅ Loaded {len(loaded_queries)} queries")
                        st.rerun()

    with tab4:
        manual_queries = st.text_area("Enter Queries (one per line)", 
                                     height=200,
                                     placeholder="What is the capital of France?\nExplain quantum computing.",
                                     key="manual_queries")
        col9, col10 = st.columns([1,1])
        with col9:
            if st.button("➕ Add Manual Queries", type="primary"):
                if manual_queries.strip():
                    new_queries = [line.strip() for line in manual_queries.split('\n') if line.strip()]
                    st.session_state.queries.extend(new_queries)
                    st.success(f"✅ Added {len(new_queries)} queries")
        with col10:
            if st.button("🗑️ Clear All Queries"):
                st.session_state.queries = []
                st.session_state.pop('last_queries_file_id', None)
                st.success("✅ Cleared")
                st.rerun()

    if st.session_state.queries:
        st.success(f"**Loaded: {len(st.session_state.queries)} queries**")
        with st.expander("Edit Queries", expanded=False):
            queries_to_delete = []
            for i, query in enumerate(st.session_state.queries):
                col_c, col_d = st.columns([5,1])
                with col_c:
                    edited = st.text_area(f"Query {i+1}", query, height=60, key=f"edit_query_{i}")
                    if edited != query:
                        st.session_state.queries[i] = edited
                with col_d:
                    if st.button("🗑️", key=f"del_query_{i}"):
                        queries_to_delete.append(i)
            for idx in sorted(queries_to_delete, reverse=True):
                st.session_state.queries.pop(idx)
            if queries_to_delete:
                st.rerun()
    else:
        st.warning("⚠️ No queries loaded")

    st.divider()

    # Run Tests
    has_prompts = bool(st.session_state.yaml_variations or st.session_state.system_prompts)
    if st.session_state.analysis and has_prompts and st.session_state.queries:
        st.subheader("🚀 Run Tests")

        specs = st.session_state.analysis.spec
        api_specs = [s for s in specs if s.is_api]
        func_specs = [s for s in specs if not s.is_api]

        selected_spec = None
        test_label = ""

        # Select endpoint/function to test
        if api_specs:
            options = [f"{s.http_method} {s.additional_params.get('route_path', s.api_endpoint)}" for s in api_specs]
            selected = st.selectbox("🌐 Select API Endpoint to test:", options)
            selected_spec = api_specs[options.index(selected)]
            test_label = f"API Endpoint: {selected}"
        elif func_specs:
            # Check if any spec is a script
            script_specs = [s for s in func_specs if s.execution_type == "script"]
            other_funcs = [s for s in func_specs if s.execution_type != "script"]
            
            if script_specs:
                # Script-based execution (CrewAI, etc.)
                st.info("🎬 **Script Mode Detected**: Will execute code with dynamically generated YAML")
                selected_spec = script_specs[0]
                test_label = "Script Execution (CrewAI/Agent Framework)"
            elif other_funcs:
                likely_prompt_funcs = []
                for f in other_funcs:
                    if f.system_prompt_param or (f.function_name and re.search(r"prompt|system", f.function_name, re.I)):
                        likely_prompt_funcs.append(f)

                if len(likely_prompt_funcs) > 1:
                    options = [f"{s.function_name}()" for s in likely_prompt_funcs]
                    selected = st.selectbox("🔧 Select Function to test:", options)
                    selected_spec = likely_prompt_funcs[options.index(selected)]
                    test_label = f"Function: {selected}"
                elif len(likely_prompt_funcs) == 1:
                    selected_spec = likely_prompt_funcs[0]
                    test_label = f"Function: {likely_prompt_funcs[0].function_name}()"
                else:
                    st.warning("⚠️ No function directly linked to system prompt detected")
                    selected_spec = None
                    test_label = "All Functions"
            else:
                selected_spec = None
                test_label = "Mock Test"
        else:
            st.warning("⚠️ No valid API endpoints or functions detected for testing.")
            return

        # Calculate total tests
        if st.session_state.yaml_variations:
            num_prompts = len(st.session_state.yaml_variations)
            prompt_type = "YAML variations"
        else:
            num_prompts = len(st.session_state.system_prompts)
            prompt_type = "prompts"
        
        total_tests = num_prompts * len(st.session_state.queries)
        st.info(f"**Ready: {total_tests} tests** ({num_prompts} {prompt_type} × {len(st.session_state.queries)} queries) on {test_label}")

        # Smart merge option (only if no system prompt param)
        if selected_spec and not selected_spec.system_prompt_param:
            st.warning(f"⚠️ The selected endpoint/function (`{test_label}`) "
                    f"does not explicitly accept a system prompt parameter.")
            st.info("You can either proceed using only the query, "
                    "or smartly merge the system prompt and query together before sending.")

            st.session_state.smart_merge_mode = st.radio(
                "Choose how to handle missing system prompt param:",
                ["Proceed normally (query only)", "Smart merge (combine system prompt + query)"],
                key=f"merge_option_global"
            )
        else:
            st.session_state.smart_merge_mode = "Proceed normally (query only)"

        # Run Button
        col11, col12, col13 = st.columns([1, 2, 1])
        with col12:
            if st.button("▶️ Run Tests", type="primary", use_container_width=True):
                agent = PromptTestingAgent()
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                test_num = 0

                # Determine which prompts to use
                if st.session_state.yaml_variations:
                    prompt_source = st.session_state.yaml_variations
                    use_yaml_mode = True
                else:
                    prompt_source = [{"prompt": p} for p in st.session_state.system_prompts]
                    use_yaml_mode = False

                for i, prompt_variant in enumerate(prompt_source):
                    for j, query in enumerate(st.session_state.queries):
                        test_num += 1
                        status_text.text(f"Running test {test_num}/{total_tests} on {test_label}...")

                        if selected_spec:
                            # Build payload with YAML variant if available
                            if use_yaml_mode:
                                # For script execution, pass full variant as payload
                                if selected_spec.execution_type == "script":
                                    payload = deepcopy(prompt_variant)
                                    payload['query'] = query
                                    prompt_display = json.dumps(prompt_variant, indent=2)
                                else:
                                    # Use YAML variant for API/function
                                    payload = agent.create_payload(
                                        selected_spec, 
                                        system_prompt="",
                                        query=query,
                                        yaml_variant=prompt_variant
                                    )
                                    prompt_display = json.dumps(prompt_variant, indent=2)
                            else:
                                # Use traditional system prompt
                                system_prompt = prompt_variant.get("prompt", "")
                                payload = agent.create_payload(selected_spec, system_prompt, query)
                                prompt_display = system_prompt

                            # Apply smart merge if needed (not for scripts)
                            if selected_spec.execution_type != "script" and \
                               (not selected_spec.system_prompt_param) and \
                               (st.session_state.smart_merge_mode == "Smart merge (combine system prompt + query)") and \
                               selected_spec.query_param:
                                if use_yaml_mode:
                                    merged_parts = [f"{k}: {v}" for k, v in prompt_variant.items()]
                                    merged_text = "\n".join(merged_parts) + f"\n\nUser Query:\n{query}"
                                else:
                                    merged_text = f"System Prompt:\n{prompt_display}\n\nUser Query:\n{query}"
                                payload[selected_spec.query_param] = merged_text

                            exec_result = agent.execute(selected_spec, payload, code_content=st.session_state.code_content)
                            target_name = test_label
                        else:
                            # No spec selected - skip
                            if use_yaml_mode:
                                payload = deepcopy(prompt_variant)
                                payload['query'] = query
                                prompt_display = json.dumps(prompt_variant, indent=2)
                            else:
                                payload = {"system_prompt": prompt_variant.get("prompt", ""), "query": query}
                                prompt_display = prompt_variant.get("prompt", "")
                            
                            exec_result = {"success": False, "error": "No specific function or endpoint selected"}
                            target_name = "Skipped"

                        # Extract response
                        if exec_result.get("success"):
                            res = exec_result.get("result")
                            if isinstance(res, dict):
                                response_display = next(
                                    (str(res[k]) for k in ["response", "answer", "text", "result", "output"] if k in res),
                                    json.dumps(res, default=str)
                                )
                            else:
                                response_display = str(res)
                        else:
                            response_display = f"ERROR: {exec_result.get('error', 'Unknown error')}"

                        # Store result
                        result_entry = {
                            'Target': target_name,
                            'Prompt Variation #': i + 1,
                            'Query': query,
                            'Response': response_display[:500] + "..." if len(response_display) > 500 else response_display,
                            'Full Response': response_display,
                            'Payload Sent': payload,
                            'Exec Raw': exec_result
                        }
                        
                        # Add prompt details based on mode
                        if use_yaml_mode:
                            result_entry['YAML Variant'] = prompt_display
                            # Add individual fields as columns
                            for key, val in prompt_variant.items():
                                result_entry[f'YAML_{key}'] = str(val)[:100] + "..." if len(str(val)) > 100 else str(val)
                        else:
                            result_entry['System Prompt'] = prompt_display[:100] + "..." if len(prompt_display) > 100 else prompt_display
                        
                        results.append(result_entry)
                        progress_bar.progress(test_num / total_tests)

                status_text.text("✅ All tests completed!")
                df = pd.DataFrame(results)
                st.session_state.test_results = df

        # Display Results
        if st.session_state.test_results is not None:
            st.divider()
            st.subheader("📊 Test Results")

            df = st.session_state.test_results
            
            # Display columns based on what's available
            display_cols = ['Target', 'Prompt Variation #', 'Query', 'Response']
            if 'YAML Variant' in df.columns:
                display_cols.insert(2, 'YAML Variant')
            elif 'System Prompt' in df.columns:
                display_cols.insert(2, 'System Prompt')
            
            st.dataframe(df[display_cols], use_container_width=True, height=400)

            # Summary statistics
            st.markdown("### 📈 Summary")
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.metric("Total Tests", len(df))
            with col_s2:
                success_count = df['Exec Raw'].apply(lambda x: x.get('success', False) if isinstance(x, dict) else False).sum()
                st.metric("Successful", success_count)
            with col_s3:
                error_count = len(df) - success_count
                st.metric("Errors", error_count)

            # Download options
            col1, col2, col3 = st.columns(3)
            with col1:
                csv = df.to_csv(index=False)
                st.download_button("📥 Download CSV", csv, "prompt_test_results.csv", "text/csv", use_container_width=True)
            with col2:
                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Results')
                excel_buffer.seek(0)
                st.download_button("📥 Download Excel", excel_buffer, "prompt_test_results.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
            with col3:
                if st.button("🧹 Clear Results", use_container_width=True):
                    st.session_state.test_results = None
                    st.success("✅ Results cleared")
                    st.rerun()

            # Detailed view
            with st.expander("🔍 View Detailed Results", expanded=False):
                for idx, row in df.iterrows():
                    st.markdown(f"### Test {idx + 1}")
                    st.markdown(f"**Target:** {row['Target']}")
                    st.markdown(f"**Query:** {row['Query']}")
                    
                    if 'YAML Variant' in row:
                        st.markdown("**YAML Variant:**")
                        st.code(row['YAML Variant'], language='yaml')
                    elif 'System Prompt' in row:
                        st.markdown(f"**System Prompt:** {row['System Prompt']}")
                    
                    st.markdown("**Response:**")
                    st.text_area("Response", row['Full Response'], height=150, key=f"response_{idx}", disabled=True)
                    
                    st.markdown("**Payload Sent:**")
                    st.json(row['Payload Sent'])
                    
                    if isinstance(row['Exec Raw'], dict):
                        if not row['Exec Raw'].get('success'):
                            st.error(f"Error: {row['Exec Raw'].get('error', 'Unknown')}")
                            if 'traceback' in row['Exec Raw']:
                                with st.expander("View Traceback"):
                                    st.code(row['Exec Raw']['traceback'])
                    
                    st.divider()

if __name__ == "__main__":
    main()