import os
import json
import traceback
import sys

# Force all prints to be immediate
sys.stdout.reconfigure(line_buffering=True)

print("Starting extraction...")
try:
    with open('Medicinal Leaf Classifier _Standalone_.html', 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    print("Read file of length:", len(content))
    
    start_str = '<script type="__bundler/template">'
    start_idx = content.find(start_str)
    if start_idx != -1:
        start_idx += len(start_str)
        end_idx = content.find('</script>', start_idx)
        template_json = content[start_idx:end_idx].strip()
        print("Found template JSON of length:", len(template_json))
        
        template_html = json.loads(template_json)
        with open('app_ui.html', 'w', encoding='utf-8') as out:
            out.write(template_html)
        print('SUCCESS: Created app_ui.html')
    else:
        print('Did not find the template tag.')
except Exception as e:
    print('Error:', e)
    traceback.print_exc()
