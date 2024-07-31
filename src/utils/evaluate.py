from src.config.logging import logger
from typing import Optional
from typing import Dict 
from typing import Any 


def normalize_to_float(value: Any) -> Optional[float]:
    """
    Attempts to convert a value to a float. If the value is a float string,
    it converts it to a float first. If the value is -1 or cannot be converted,
    it returns None.

    Parameters:
    value (Any): The value to be converted.

    Returns:
    Optional[float]: The float representation of the value or None if conversion fails.
    """
    try:
        float_value = float(value)
        return float_value if float_value != -1 else None
    except (ValueError, TypeError):
        return None


def compare_json_objects(json1: Dict, json2: Dict) -> bool:
    """
    Compares two JSON objects based on 'code', 'value', 'unit', and 'year' fields.

    The comparison checks if the 'code' fields are equal as strings, 
    'value' and 'year' fields are equal as floats (with normalization to None for invalid values),
    and if the 'unit' fields are equal after stripping whitespace.

    Parameters:
    json1 (Dict): The first JSON object.
    json2 (Dict): The second JSON object.

    Returns:
    bool: True if 'code', 'value', 'unit', and 'year' fields are equal, False otherwise.
    """
    try:
        code1 = str(json1.get('code', ''))
        code2 = str(json2.get('code', ''))
        value1 = normalize_to_float(json1.get('value', -1))
        value2 = normalize_to_float(json2.get('value', -1))
        year1 = normalize_to_float(json1.get('year', -1))
        year2 = normalize_to_float(json2.get('year', -1))
        unit1 = json1.get('unit', '').strip()
        unit2 = json2.get('unit', '').strip()

        logger.info(f"Extracted values - Code: {code1} vs {code2}, Value: {value1} vs {value2}, Year: {year1} vs {year2}, Unit: '{unit1}' vs '{unit2}'")

        # Ideal - Compare against 4 dimensions
        # result = code1 == code2, value1 == value2, year1 == year2, unit1 == unit2 
        result = code1 == code2, value1 == value2 
        logger.info(f"Comparison result: {result}")
        
        return result
    except ValueError as e:
        logger.error(f"Error converting 'value' or 'year' to float: {e}")
        return False

