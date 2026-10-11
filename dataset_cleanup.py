import re
import mwparserfromhell

def cleanup_wikimedia_text(text):
    wikicode = mwparserfromhell.parse(text)
    text = wikicode.strip_code()
    # Remove Wikipedia-specific artifact artifacts & boilerplates
    # (e.g., "See also", "References", "External links" sections at the end)
    text = re.split(r'\n==\s*(See also|References|External links|Notes|Bibliography)\s*==', text, flags=re.IGNORECASE)[0]
    # Remove markdown/wiki section headers (e.g., == Section ==, === Sub-section ===)
    text = re.sub(r'={2,}\s*(.*?)\s*={2,}', r'\1.', text)
    # Normalize whitespace (replace multiple spaces, tabs, and newlines with a single space/newline)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n', text)
    # Remove dangling brackets or template leftovers commonly found in raw dumps
    text = re.sub(r'\{\|[^*]*?\|\}', '', text) # Wiki tables
    text = re.sub(r'\[\[File:.*?\]\]', '', text, flags=re.IGNORECASE) # Image tags
    # Remove corrupted dangling brackets
    text = re.sub(r'\{\|.*', '', text) # Wiki tables
    return text
