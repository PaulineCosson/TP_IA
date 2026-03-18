import glob
import pathlib
import re

files = glob.glob('corpus/saga_freezer/*.txt')
print('files', len(files))

total_words = 0
for f in files:
    text = pathlib.Path(f).read_text(encoding='utf-8')
    words = re.findall(r"\w+", text)
    total_words += len(words)
    print(f, len(text), 'chars', len(words), 'words')

print('total words', total_words)
