""" IMPORTS """
import re
import unicodedata
import string


""" PRE DOCUMENT CLEANING"""

def pre_clean_text(text):
    """
    Main cleaning of a document

    Paramenters:
    - text (str): The input of the text to be cleaned

    Returns:
    - str: The cleaned text
    """

    clean_text = map_special_characters(text,convert_greek_to_names=True)
    clean_text,_ = remove_links(clean_text)
    clean_text = clean_text_characters(clean_text)
    clean_text = replace_abbreviations(clean_text)
    clean_text = clean_text_patterns(clean_text)
    clean_text = custom_cleaning_common_patterns(clean_text)
    clean_text = replace_comparison_symbols(clean_text)
    clean_text = re.sub(r' +',' ',clean_text)
    clean_text = re.sub(r'\n{3,}','\n\n',clean_text)
    return clean_text

def custom_cleaning_common_patterns(text):

    # Regular expression to match "-Modelle" or "-Modellen" (case-insensitive)
    pattern = r"-(modelle|modellen|modell)"

    # Replace with "Modelle" or "Modellen" (removing the hyphen)
    cleaned_text = re.sub(pattern, r" \1", text,flags=re.IGNORECASE)
    
    return cleaned_text

def map_special_characters(text, convert_greek_to_names=False):
    """
    Cleaning of special characters and conversion of strange characters
    Converts greek letters to their names

    Paramenters:
    - text (str): The input of the text to be cleaned
    - convert_greek_to_names (bool): Gives users the option to replace greek characters

    Returns:
    - str: The cleaned text
    """
    # Define specific mappings
    char_mapping = {
        "ƞ": "η",    # Map Latin n with long right leg to Greek eta
        "𝐸": "E",
        "𝐴": "A",
        "𝑇": "T",
        "ₕ": "h",
        "²": "2",
        "³": "3",
        "¹": "1",
        "ł": "l",    # Map Polish ł to l
        "𝑠": "s",    # Map styled s to regular s
        "–": "-",    # Map en dash to hyphen
        "—": "-",     # Keep equals sign as is
        "≤": "<=",   # Map less than or equal to
        "≥": ">=",    # Map greater than or equal to
        "×": "*",
        "₂": "2",
        "Ł": "L",
        "§§": " Paragrafen ",
        "§":" Paragraf ",
        "~":"-",
        '´':"'",
        '’':"'",
        "”":'"',
        "\no ":"\n- ",
        "·":"",
        "‑":"-"
        }
    
    # Define Greek letters to names mapping (if needed)
    greek_to_name = {
        "η": "eta",
        "λ": "lambda",
        "μ": "mu",
        "θ": "theta",
        "π": "pi"
    }

    # Define characters to delete (e.g., trademark symbol)
    chars_to_delete = ["®","ₛ","™","©","“","„",'"',"|","\uf0a7"]
    
    # Apply specific character mappings
    for char, replacement in char_mapping.items():
        text = text.replace(char, replacement)

    # Remove characters specified for deletion
    for char in chars_to_delete:
        text = text.replace(char, "")
    
    # Optionally convert Greek letters to names
    if convert_greek_to_names:
        for greek, name in greek_to_name.items():
            text = text.replace(greek, name)
    
    # Normalize accents, excluding German-specific characters
    normalized_text = []
    for char in text:
        if char in "äöüßÄÖÜ":  # Keep German characters as they are
            normalized_text.append(char)
        else:
            # Decompose accented characters, keep only base letter
            normalized_char = unicodedata.normalize("NFD", char)
            normalized_char = "".join([c for c in normalized_char if unicodedata.category(c) != "Mn"])
            normalized_text.append(normalized_char)

    return "".join(normalized_text)

def remove_links(text):
    """
    Removing and extraction of links around <> in the text

    Paramenters:
    - text (str): The input of the text to be cleaned

    Returns:
    - str: The cleaned text
    """

    # Pattern to match URLs in both forms: (<http...>) or <http...>
    pattern = r'\(<(http[^\)]+?)>\)|<(http[^\>]+?)>'
    # Pattern to match any text in the format (< [some text] >)
    bracket_pattern = r'\(<[^>]*?>\)'
    
    urls_found = []
    
    # Define a replacement function
    def replace_link(match):
        url = match.group(1) if match.group(1) else match.group(2)
        urls_found.append(url)
        return ""  # Remove the matched URL from the text

    # Substitute the URLs with an empty string and capture them
    cleaned_text = re.sub(pattern, replace_link, text, flags=re.DOTALL)
    # Remove any text in the format (< [some text] >)
    cleaned_text = re.sub(bracket_pattern, "", cleaned_text)

    # Regex patterns
    # Match links within parentheses that start with 'https:'
    pattern_with_end_parenthesis = r'\(https:[^\)]*\)'
    # Match links without a closing parenthesis that go till the end of the line
    pattern_without_end_parenthesis = r'\(https:[^\)]*$'

    # Remove links that are fully enclosed in parentheses
    cleaned_text = re.sub(pattern_with_end_parenthesis, '', cleaned_text)
    
    # Remove links that start with '(https:' and extend to the end of the line
    cleaned_text = re.sub(pattern_without_end_parenthesis, '', cleaned_text)
    
    
    return cleaned_text, urls_found

def clean_text_characters(text):
    """
    Clean characters of text and correct some patterns of Pages numbers of index

    Paramenters:
    - text (str): The input of the text to be cleaned

    Returns:
    - str: The cleaned text
    """

    clean_text = re.sub(r'\.{2,}', ' Seite ', text)
    clean_text = re.sub(r' {2,}', ' ', clean_text)
    clean_text = re.sub(r'(\.\s)+\.', '', clean_text)
    clean_text = clean_text.replace("", "•")
    clean_text = clean_text.replace("□", "-")
    clean_text = clean_text.replace("(#)", "")
    clean_text = re.sub(r"(\d+(?:\.\d+)?)\n(\d+)\n\n(.+)", r"\1 \3 Seite \2", clean_text)

    return clean_text

# Dictionary of specific abbreviations and their full forms
abbreviation_dict = {
    "z.B.": "zum Beispiel",
    "z. B.": "zum Beispiel",
    "bzw": "beziehungsweise",
    "bzw.": "beziehungsweise",
    "i. V. m.": "in Verbindung mit",
    "i. S. v.": "im Sinne von",
    "ggf.": "gegebenenfalls",
    "Abs.": "Absatz",
    "Nr.": "Nummer",
    "bspw.": "beispielsweise",
    "d.h.": "das heißt",
    "D. h.": "Das heißt",
    "d. h.": "das heißt",
    "einschl.": "einschließlich",
    "e.V.": "eingetragener Verein",
    "inkl.": "inklusive",
    "gem.": "gemäß",
    "o. g.": "oben genannt",
    "sog.": "sogenannt",
    "s.u.": "siehe unten",
    "u.a.": "unter anderem",
    "bzgl.":"bezüglich"
}

def replace_abbreviations(text, abbreviation_dict = abbreviation_dict):
    """
    Clean common abbreviations used in the german language

    Paramenters:
    - text (str): The input of the text to be cleaned
    - abbreviation_dict (dict): Dictionary with abbreviations. Key is abbreviation, value is the replacement.

    Returns:
    - str: The cleaned text
    """
    sorted_abbreviations = sorted(abbreviation_dict.keys(), key=len, reverse=True)
    
    for abbr in sorted_abbreviations:
        # Modify the regex pattern to match word boundaries or certain punctuation around the abbreviation
        text = re.sub(rf'(?<!\w){re.escape(abbr)}(?!\w)', abbreviation_dict[abbr], text)
    
    return text



def replace_comparison_symbols(text):
    """
    Replace comparison symbols to text, helps for better understanding to models

    Paramenters:
    - text (str): The input of the text to be cleaned

    Returns:
    - str: The cleaned text
    """
    # Define patterns to handle both standard and reversed orders of symbols
    replacements = {
        r'>=': ' größer oder gleich ',
        r'=>': ' größer oder gleich ',
        r'<=': ' kleiner oder gleich ',
        r'=<': ' kleiner oder gleich ',
        r'>': ' größer als ',
        r'<': ' kleiner als ',
        r'=': ' gleich '  # Optional: add handling for standalone "=" if needed
    }
    
    # Replace each symbol pattern in the text with its German phrase equivalent
    for symbol, german_phrase in replacements.items():
        text = re.sub(re.escape(symbol), german_phrase, text)
    
    return text

patterns_remove = [
    {"type": "contains", "value": "Folgende FAQ wurden bei"},
    {"type": "contains", "value": "Folgende FAQ wurden im"},
    {"type": "contains", "value": "Folgende FAQ wurde bei"},
    {"type": "contains", "value": "Restrukturierung der FAQ"},
    {"type": "contains", "value": "Ältere FAQ-Versionen"},
    {"type": "contains", "value": "BEG FAQ Stand:"},
    {"type": "contains", "value": "FAQ im Änderungsmodus"},
    {"type": "contains", "value": "Haben Sie eine bestimmte Frage?"},
    {"type": "contains", "value": "AdobeStock/ytemha34"},
    {"type": "contains", "value": "Gebärdensprache"},
    {"type": "contains", "value": "Barrierefreiheit"},
    {"type": "contains", "value": "Leichte Sprache"},
    {"type": "contains", "value": "Impressum"},
    {"type": "contains", "value": "Datenschutz"},
    {"type": "contains", "value": "Nutzungsbedingungen"},
    {"type": "contains", "value": "Öffnet PDF"},
    {"type": "contains", "value": "Öffnet Einzelsicht"}
]

def clean_text_patterns(text, patterns=patterns_remove):
    """
    Removes lines from the text based on specified patterns.
    
    Parameters:
    - text (str): The input text to be cleaned.
    - patterns (list of dict): A list of dictionaries where each dictionary has:
        - 'type' (str): Specifies if the pattern should be matched at the 'start', 'contains', or 'exact' of the line.
        - 'value' (str): The substring to be matched for deletion.

    Returns:
    - str: The cleaned text.
    """
    # Split text into lines
    lines = text.splitlines()
    
    # Filter lines based on patterns
    cleaned_lines = []
    for line in lines:
        delete_line = False
        for pattern in patterns:
            if pattern['type'] == 'start' and line.startswith(pattern['value']):
                delete_line = True
                break
            elif pattern['type'] == 'contains' and pattern['value'] in line:
                delete_line = True
                break
            elif pattern['type'] == 'exact' and line == pattern['value']:
                delete_line = True
                break
        if not delete_line:
            cleaned_lines.append(line)
    
    # Join the cleaned lines back into a single string
    return "\n".join(cleaned_lines)


""" POST DOCUMENT CLEANING - BM 25 VOCABULARY EXTRACTION"""

def query_clean(text):

    clean_text = ''
    extracted_dict = {}

    return clean_text,extracted_dict


def post_clean_document(text):
    """
    Post cleaning of a document to extract later the BM 25 vocabulary

    Paramenters:
    - text (str): The input of the text to be cleaned

    Returns:
    - str: The cleaned text
    - dict: Dictionary with all the values that were extracted from the text, ie. norms, values, dates, etc
    """

    extracted_dict={}
    values_list = []
    cleaner_text,dates = extract_and_remove_german_dates(text)
    extracted_units,cleaner_text = extract_units(cleaner_text)
    for unit,values in extracted_units.items():
        _,labels_numbers = categorize_values(values,bins_dict,unit)
        values_list += labels_numbers
    cleaner_text,nummers_buchstaben = extract_numbers_letters(cleaner_text)
    numbers_list,letters_list = expand_numbers_letters(nummers_buchstaben)
    cleaner_text,numbers,letters = extract_sections_and_letters(cleaner_text)
    cleaner_text,norms = find_norms(cleaner_text)
    extracted_dict['numbers'] = numbers_list+numbers
    extracted_dict['letters'] = letters_list+letters
    extracted_dict['dates'] = dates
    extracted_dict['norms'] = norms
    cleaner_text = remove_websites(cleaner_text)
    cleaner_text = clean_characters_post(cleaner_text)
    extracted_ranges, cleaner_text = extract_ranges(cleaner_text)
    for unit,values in extracted_ranges.items():
        _,labels_numbers = categorize_values(values,bins_dict,unit)
        values_list += labels_numbers
    cleaner_text = compound_word_fixing(cleaner_text)
    cleaner_text, par_brack_terms = extract_and_clean_text(cleaner_text)
    extracted_dict['values'] = values_list
    extracted_dict['parentheses_terms'] = par_brack_terms
    cleaner_text = remove_numbers_alone(cleaner_text)

    return cleaner_text,extracted_dict

def extract_units(text):
    """
    Extract values of specific measurement units, i.e. kW, W/(mK), etc.

    Paramenters:
    - text (str): The input of the text to be cleaned

    Returns:
    - dict: Dictionary with all the values converted to floats. Every key has a list with the falues:
        - keys (str): Measurement unit
        - values (float): List with extracted values
    - str: The cleaned text
    """
    # Define the measurement units to search for
    units = ["kW", "W/(m2K)", "W/(mK)", "lm/W", "mg/m3", "Nm", "N", "°C", "m2", "m3", "%", "kg", "€/m2","W/(m3/h)","kWh/(m2a)","kWh/m2"]
    # Store results in a dictionary
    extracted_units = {}

    for unit in units:
        # Build the regex pattern
        pattern = rf"\b(\d{{1,3}}(?:[.,]?\d{{3}})*[.,]?\d*)\s?{re.escape(unit)}(?:[\s.,;*:)]|$)"
        # Find all matches
        matches = re.findall(pattern, text)
        
        # Convert matched numbers to float
        float_values = []
        for match in matches:
            # If there are both dot and comma, determine the format
            if "," in match and "." in match:
                # Remove all dots (assumed to be thousand separators) and convert last comma to dot
                normalized_number = match.replace(".", "").replace(",", ".", 1)
            else:
                # If there's only one symbol, treat it as the decimal
                normalized_number = match.replace(",", ".")
            
            # Convert to float
            try:
                float_values.append(float(normalized_number))
            except ValueError:
                print(f"Skipping invalid number format: {match}")

        # Store float values in dictionary under the unit key
        extracted_units[unit] = float_values

        # Build a substitution pattern to remove all matched text for this unit
        text = re.sub(rf"\b(\d{{1,3}}(?:[.,]?\d{{3}})*[.,]?\d*)\s?{re.escape(unit)}(?:[\s.,;*:)]|$)", "", text)

    return extracted_units, text

def extract_ranges(text):

    # Define units to search for
    units = ["kW", "W/(m2K)", "W/(mK)", "lm/W", "mg/m3", "Nm", "N", "°C", "m2", "m3", "%", "kg", "€/m2"]
    extracted_ranges = {}  # Dictionary to store results by unit

    for unit in units:
        # Regex pattern for finding ranges with "bis" or "-" separators
        pattern = rf"\s\(?({re.escape(unit)})\)?\s(\d{{1,3}}(?:[.,]\d{{1,3}})*)\s?(bis|-)?\s?(\d{{1,3}}(?:[.,]\d{{1,3}})*)?"
        matches = re.findall(pattern, text)
        float_values = []  # List to store ranges or single values for each unit

        for match in matches:
            unit, start, separator, end = match
            # unit = unit.replace('(','').replace(')','')
            
            # Normalize start number
            if "," in start and "." in start:
                start = start.replace(".", "").replace(",", ".", 1)
            else:
                start = start.replace(",", ".")
            start_value = float(start)
            
            # Normalize end number if it exists
            if end:
                if "," in end and "." in end:
                    end = end.replace(".", "").replace(",", ".", 1)
                else:
                    end = end.replace(",", ".")
                end_value = float(end)
                float_values.append((start_value, end_value))  # Store as a range tuple
            else:
                float_values.append((start_value,))  # Store single value as a tuple for consistency

        # Store float values in dictionary under the unit key
        if float_values:
            extracted_ranges[unit] = float_values

        # Remove matched text from the original text
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    return extracted_ranges, text

def transform_ranges_to_individual_values(extracted_ranges):
    # Initialize a new dictionary to store individual values
    individual_values = {}

    for unit, ranges in extracted_ranges.items():
        # Flatten each tuple in the list into individual values
        flat_values = []
        for range_tuple in ranges:
            flat_values.extend(range_tuple)  # Add each value from the tuple individually

        # Store the flattened list under the unit key
        individual_values[unit] = flat_values

    return individual_values

bins_dict = {
    "kW": [(0, 2), (2, 3),(3,4),(4,5),(5,7.5),(7.5,10),(10,15),(15, 20),(20,25),(25,37.5),(37.5,50), (50, 100), (100, 250), (250, 500), (500,1000),(1000, float('inf'))],
    "°C": [(0, 15),(15, 35),35,(35, 55),55,(55, float('inf'))],
    "mg/m3":[(0,0.3),(0.3,0.75),(0.75,1),(1,2),(2,5),(5,10),(10,20),(20,50),(50,100),(100,500),(500,1000),(1000,float('inf'))],
    "%":[(0,25),(25,50),(50,75),(75,100),(100,125),(125,150),(150,175),(175,200),(200,250),(250,float('inf'))],
    "W/(m2K)":[(0,1),(1,float('inf'))]    
}

def categorize_values(values, bins_dict = bins_dict, unit="kW"):
    """
    Categorizes values based on bins defined in bins_dict. Handles single-value bins directly in bins_dict.
    
    Parameters:
    - values (list): List of values to categorize.
    - bins_dict (dict): Dictionary of bins by unit.
    - unit (str): The measurement unit for categorization (default: "kW").
    """
    values = [item for sublist in values for item in (sublist if isinstance(sublist, tuple) else (sublist,))]

    # Check if bins exist for the specified unit
    if unit in bins_dict:
        bins = bins_dict[unit]
        
        # Automatically create labels from bins
        labels = []
        formatted_bins = []
        for entry in bins:
            # Check if entry is a tuple (range) or a single value
            if isinstance(entry, tuple):
                lower, upper = entry
                formatted_bins.append((lower, upper))
                if upper == float('inf'):
                    labels.append(f"{int(lower)}+")
                else:
                    labels.append(f"{int(lower)}-{int(upper)}")
            else:
                # Single value bin
                formatted_bins.append((entry, entry))
                labels.append(f"{int(entry)}")
        
        # Result lists
        categorized_values = []
        categorized_labels = []
        
        # Categorize each value
        for value in values:
            # Check each formatted bin
            for i, (lower, upper) in enumerate(formatted_bins):
                if lower <= value < upper or (lower == upper == value):
                    categorized_values.append(value)
                    categorized_labels.append(f"{labels[i]} {unit}")
                    break  # Stop after finding the correct bin

    else:
        # If no bins are specified, approximate to the nearest integer
        categorized_values = [round(value) for value in values]
        categorized_labels = [f"{unit}" for value in values]
    
    return categorized_values, categorized_labels


def extract_numbers_letters(text):

    initial_pattern = r'(Nummer[n]?)\s+((?:\d+\.)+\d+)'
    extending_pattern = r'^(,\s*|und\s+|oder\s+|sowie\s+|bis\s+|Buchstabe\s+|Nummer\s+)'
    number_or_letter_pattern = r'^\s*(\d+(?:\.\d+)*|[a-zA-Z](?=\s|[.,;]|$)|Nummer\s+\d+(?:\.\d+)*|Buchstabe\s+[a-zA-Z])'

    # Find all initial occurrences of the pattern
    end = 0
    matches = []
    for match in re.finditer(initial_pattern, text):
        # Start with the first "Nummer/n" pattern found
        if end >= match.end():
            continue
        start = match.start()
        end = match.end()
        extracted = match.group(0)
        while True:
            original_slice = text[end:]
            next_text = original_slice.lstrip()
            # Check if .strip() removed any leading whitespace
            whitespace_removed = len(original_slice) - len(next_text)
            add_whitespace = 0
            if whitespace_removed > 0:
                add_whitespace = whitespace_removed
            extension_match = re.match(extending_pattern, next_text)
            if extension_match:
                extension = extension_match.group(0)
                next_text2 = next_text[extension_match.end():]
                number_letter_match = re.match(number_or_letter_pattern,next_text2)
                if number_letter_match:
                    end += number_letter_match.end() + extension_match.end() + add_whitespace
                else:
                    break
            else:
                break
        text_found = text[start:end].strip()
        matches.append(text_found)

    # Remove all matched sections from text by replacing each match with an empty string
    modified_text = text
    for match in matches:
        modified_text = re.sub(re.escape(match), '', modified_text, count=1)

    return modified_text,matches 

def expand_numbers_letters(text_list):
    
    expanded_numbers_total = []
    expanded_letters_total = []
    for text in text_list:
        text = re.sub('Nummern','Nummer',text)
        # Add Nummer when missing // Add Buchstabe when missing
        text_split = text.split()
        new_text_split = []
        for n,i in enumerate(text_split):
            number = re.match(r'\b\d+(?:\.\d+)*\b',i)
            if number:
                if n >= 0:
                    prev_word = text_split[n-1]
                    if prev_word != "Nummer":
                        word_i = f"Nummer {i}"
                    else:
                        word_i = i
                else:
                    word_i = f"Nummer {i}"
                new_text_split.append(word_i)
            else:
                new_text_split.append(i)
        text = " ".join(new_text_split)
        text = re.sub(r'(?<!Buchstabe )(\b[a-z]\b)', r'Buchstabe \1', text)

        text = re.sub(r'\b(?:oder|und|sowie)\b','',text)
        text = re.sub(r'\,','',text)
        text = re.sub(r'\s{2,}',' ',text)

        matches = re.findall(r'(Nummer \d+(?:\.\d+)*|Buchstabe [a-z]|[a-zA-Z]+)', text)

        expanded_list = []
        i = 0
        while i < len(matches):
            # Check for "bis" to determine if we have a range
            if matches[i] == "bis" and i > 0 and i < len(matches) - 1:
                start = matches[i - 1]
                end = matches[i + 1]
                
                # Check if both start and end are of the same type (e.g., both "Nummer" or both "Buchstabe")
                if start.startswith("Nummer") and end.startswith("Nummer"):
                    # Extract the numbers for the range
                    start_num = re.search(r'(\d+(?:\.\d+)*)$', start).group()
                    end_num = re.search(r'(\d+(?:\.\d+)*)$', end).group()
                    
                    # Expand for "Nummer" range and add to expanded list
                    expanded_list.extend([f"Nummer {n}" for n in expand_number_range(start_num, end_num)])
                elif start.startswith("Buchstabe") and end.startswith("Buchstabe"):
                    # Extract letters for the range
                    start_letter = start.split()[-1]
                    end_letter = end.split()[-1]
                    
                    # Expand for "Buchstabe" range and add to expanded list
                    expanded_list.extend([f"Buchstabe {char}" for char in expand_letter_range(start_letter, end_letter)])
                
                # Skip the start, "bis", and end items, since they are now expanded
                i += 2
            else:
                # Only add if the current item isn't part of a range
                # If "bis" comes afterward, we will skip this item during expansion
                if (i < len(matches) - 1 and matches[i + 1] != "bis") or i == len(matches) - 1:
                    expanded_list.append(matches[i])
                i += 1
        
        expanded_letters = [i for i in expanded_list if i.strip().startswith("Buchstabe")]
        expanded_numbers = [i for i in expanded_list if i.strip().startswith("Nummer")]
        expanded_letters_total.extend(expanded_letters)
        expanded_numbers_total.extend(expanded_numbers)

        

    return expanded_numbers_total,expanded_letters_total

def expand_number_range(start, end):
    # Split numbers into parts based on dots
    start_parts = list(map(int, start.split('.')))
    end_parts = list(map(int, end.split('.')))
    
    # Check if both start and end have the same "level" and matching prefix
    if len(start_parts) != len(end_parts):
        # Different levels, return both without expansion
        return [start, end]
    
    # Ensure all but the last parts are the same for expansion to be valid
    if start_parts[:-1] != end_parts[:-1]:
        # Prefix doesn't match, return both without expansion
        return [start, end]
    
    # If valid for expansion, expand only the last part
    expanded = [
        '.'.join(map(str, start_parts[:-1] + [i])) 
        for i in range(start_parts[-1], end_parts[-1] + 1)
    ]
    
    return expanded
def expand_letter_range(start, end):
    # Expand letters in the range
    return list(string.ascii_lowercase[string.ascii_lowercase.index(start):string.ascii_lowercase.index(end) + 1])


def extract_sections_and_letters(text):
    # Pattern for section numbers like #, #.#, or #.#.# at the start of a line
    # section_pattern = r'^(?:\d{1,2}(?:\.\d+){0,2})(?=\s|$)'  # Matches 1, 1.2, or 1.2.3
    section_pattern = r'^(?:\d{1,2}(?:\.\d{1,2}){0,2})(?=\s|$)(?!\.\d{4}\b)'
    # Pattern for letters with parentheses like a), b), c) at the start of a line
    letter_pattern = r'^[a-zA-Z]\)'
    # Pattern to match "Nummer: X.XX" format at the start of a line
    nummer_pattern = r'^Nummer:\s(\d+\.\d{2})'  # Matches "Nummer: X.XX"

    # Extract, format, and remove section numbers
    section_matches = re.findall(section_pattern, text, re.MULTILINE)
    formatted_sections = [f"Nummer {num}" for num in section_matches]
    text = re.sub(section_pattern, '', text, flags=re.MULTILINE)

    # Extract, format, and remove letter markers
    letter_matches = re.findall(letter_pattern, text, re.MULTILINE)
    letters = [f"Buchstabe {letter[0]}" for letter in letter_matches]
    text = re.sub(letter_pattern, '', text, flags=re.MULTILINE)

    # Extract, standardize, and remove "Nummer: X.XX" patterns
    nummer_matches = re.findall(nummer_pattern, text, re.MULTILINE)
    formatted_nummer = [f"Nummer {num}" for num in nummer_matches]
    numbers = formatted_sections + formatted_nummer
    text = re.sub(nummer_pattern, '', text, flags=re.MULTILINE)

    return text,numbers,letters

def convert_date(date_str):
    # Split the date string by dots
    day, month, year = date_str.split(".")
    month = int(month)
    if len(year) == 2:
        if int(year) > 50:
            year = f'19{year}'
        else:
            year = f'20{year}'

    # Define the German months
    months_german = [
        "Januar", "Februar", "März", "April", "Mai", "Juni",
        "Juli", "August", "September", "Oktober", "November", "Dezember"
    ]
    
    # Get the month in German using the month part of the string
    month_written = months_german[month-1]
    
    # Return in the format "Monat Jahr"
    return f"{month_written} {year}"

def extract_and_remove_german_dates(text):
    # Define regex for "DD.MM.YYYY" format
    date_numeric_pattern = r'\b(?:[0-2]?[0-9]|3[0-1])\.(?:0?[1-9]|1[0-2])\.\d{4}\b'
    
    # Define regex for "D. Monat YYYY" and "Monat YYYY" formats
    date_month_pattern = r'\b(?:(?:([1-2]?[0-9]|3[0-1])\.\s*)?(Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember))\s+(\d{4})\b'
    
    # Month name to numeric mapping
    month_to_number = {
        "Januar": "01", "Februar": "02", "März": "03", "April": "04", "Mai": "05", "Juni": "06",
        "Juli": "07", "August": "08", "September": "09", "Oktober": "10", "November": "11", "Dezember": "12"
    }
    
    # Find and extract dates in "DD.MM.YYYY" format
    numeric_dates = re.findall(date_numeric_pattern, text)
    
    # Find and extract dates in "D. Monat YYYY" and "Monat YYYY" formats
    month_dates = re.findall(date_month_pattern, text)
    
    # Convert "D. Monat YYYY" and "Monat YYYY" format dates to "DD.MM.YYYY"
    converted_month_dates = [
        f"{int(day):02}.{month_to_number[month]}.{year}" if day else f"01.{month_to_number[month]}.{year}"
        for day, month, year in month_dates
    ]
    
    # Combine all extracted dates
    all_dates = numeric_dates + converted_month_dates

    all_dates = [convert_date(i) for i in all_dates]
    
    # Create a pattern to match both date formats in the original text
    combined_pattern = f"({date_numeric_pattern})|({date_month_pattern})"
    
    # Remove all matches from the text
    cleaned_text = re.sub(combined_pattern, '', text)
    
    # Clean up any extra whitespace from removed dates
    cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text).strip()
    
    return cleaned_text, all_dates

def find_norms(text):
    """
    Identifies norms like DIN, ISO, EN, etc., with their associated names, numbers, and variants.
    
    Parameters:
    - text: str, input text to search for norms.
    
    Returns:
    - List of norms found in the text.
    """
    # Regular expression pattern to capture norms
    norm_pattern = r'\b(?:DIN|ISO|EN)(?:\s+[A-Z]{1,3})?\s+\d{3,5}(?:[-‑]\d+)?\b'
    norm_pattern = r'\b(?:DIN|ISO|EN)(?:\s+(?:ISO|IEC))?(?:\s+[A-Z]{1,3})?\s+\d{3,5}(?:[-‑]\d+)?(?:\s+Beiblatt\s+\d+)?\b'
    norm_pattern = r'\b(DIN(?:\s+EN)?(?:\s+ISO(?:/IEC)?)?(?:\s+[A-Z])?\s+\d{3,5}(?:[-‑]\d+)?(?:\s+Beiblatt\s+\d+)?)\b'
    norm_pattern = r'\b(DIN(?:\s+EN)?(?:\s+ISO(?:/IEC)?)?(?:\s+[A-Z])?\s+\d{3,5}(?:[-‑]\d+)?(?:\s+Beiblatt\s+\d+)?(?:\s+\d+)?)\b'
    
    # Find all matches of the pattern in the text
    norms = re.findall(norm_pattern, text, re.IGNORECASE)
    for term in norms:
        text = text.replace(term,'')
    
    return text,norms

def remove_websites(text):
    # Pattern 1: Starts with "http" or "www" and ends at space or line break
    text = re.sub(r'(http\S+|www\S+)(\s|\n|$)', '', text)

    # Pattern 2: Ends with .html, .de, .html/, or .de/, erase backward until a space
    text = re.sub(r'(\S+\.(com|de|info|html)(/\S*)?)(?=\s|$)', '', text)

    return text

def clean_characters_post(text):
    # Define patterns to remove specific characters
    patterns = [r':', r';', r'\*', r'•',r'#',r'\?',r'‡',r'†',r'\\',r'…',r'|',r'\!',r'\_']  # Simplify and add patterns directly

#‡\\|…\?”\(\)\]

    for patt in patterns:
        text = re.sub(patt, "", text)

    # Remove hyphens at the start of each line
    clean_text = re.sub(r'(?m)^\-', '', text)  # (?m) activates MULTILINE mode for '^'

    return clean_text

def compound_word_fixing(text):

    clean_text = re.sub(pattern=r'\/\-|\-\/',repl=' ',string =text)

    clean_text = re.sub(pattern=r'\/',repl=' ',string =clean_text)

    clean_text = re.sub(r' {2,}', ' ', clean_text)

    clean_text = re.sub(r'^[ \t]+', '', clean_text, flags=re.MULTILINE)

    return clean_text

def extract_and_clean_text(text):
    # Extract terms within parentheses or brackets
    extracted_terms = re.findall(r'\((.*?)\)|\[(.*?)\]', text)
    # Flatten list of extracted terms, ignoring None values
    extracted_terms = [term for group in extracted_terms for term in group if term]
    extracted_terms = [term for term in extracted_terms if (len(term)>2 and re.match(r'[a-zA-Z]',term))]


    # Remove all occurrences of text within parentheses or brackets from the original text
    clean_text = re.sub(r'\(.*?\)|\[.*?\]', '', text).strip()

    return clean_text, extracted_terms

def remove_numbers_alone(text):
    """
    Clean text by removing alone characters, specially numbers.

    params:
    - text (str): Text to be cleaned

    Returns:
    - str: The cleaned text with alone numbers removed
    """
    # Regular expression to match numbers that are alone
    clean_text = re.sub(r'(?<![-\w])\b\d+\b(?![-\w])', '', text)
    
    # Remove any extra spaces left after removing numbers
    clean_text = re.sub(r'\s{2,}', ' ', clean_text).strip()
    
    return clean_text