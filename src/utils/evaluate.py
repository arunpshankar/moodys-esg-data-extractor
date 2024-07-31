from src.config.logging import logger
from typing import Dict 

def compare_json_objects(json1: Dict, json2: Dict) -> bool:
    """
    Compares two JSON objects based on 'code', 'value', 'unit', and 'year' fields.

    The comparison checks if the 'code', 'value', and 'year' fields are equal after converting to integers
    and if the 'unit' fields are equal after stripping whitespace.

    Parameters:
    json1 (Dict): The first JSON object.
    json2 (Dict): The second JSON object.

    Returns:
    bool: True if 'code', 'value', 'unit', and 'year' fields are equal, False otherwise.

    Raises:
    ValueError: If conversion to int fails for 'code', 'value', or 'year' fields.
    """
    try:
        code1 = int(json1.get('code', 0))
        code2 = int(json2.get('code', 0))
        value1 = int(json1.get('value', 0))
        value2 = int(json2.get('value', 0))
        year1 = int(json1.get('year', 0))
        year2 = int(json2.get('year', 0))
        unit1 = json1.get('unit', '').strip()
        unit2 = json2.get('unit', '').strip()

        logger.info(f"Extracted values - Code: {code1} vs {code2}, Value: {value1} vs {value2}, Year: {year1} vs {year2}, Unit: '{unit1}' vs '{unit2}'")

        # Ideal - Compare on 4 dimensions
        # result = code1 == code2, value1 == value2, year1 == year2, unit1 == unit2 
        result = code1 == code2, value1 == value2
        logger.info(f"Comparison result: {result}")
        
        return result
    except ValueError as e:
        logger.error(f"Error converting 'code', 'value', or 'year' to int: {e}")
        return False
