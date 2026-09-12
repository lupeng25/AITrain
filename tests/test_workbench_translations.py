"""工作台静态文案与英文资源必须同时更新，格式占位符不能丢失。"""
from pathlib import Path
import json
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
LITERAL = re.compile(r'(QStringLiteral|tr|uiText)\(\s*((?:"(?:[^"\\]|\\.)*"\s*)+)\)')
CHINESE = re.compile(r'[\u4e00-\u9fff]')


def test_workbench_english_covers_static_text_and_preserves_placeholders():
    contexts = {}
    for context in ET.parse(ROOT / 'src/app/translations/aitrain_en_US.ts').findall('context'):
        entries = contexts.setdefault(context.findtext('name'), {})
        for message in context.findall('message'):
            source = message.findtext('source') or ''
            translation = message.find('translation')
            assert translation is not None and translation.get('type') not in ('unfinished', 'obsolete', 'vanished'), source
            target = translation.text or ''
            assert target, source
            assert sorted(re.findall(r'%[1-9][0-9]*', source)) == sorted(re.findall(r'%[1-9][0-9]*', target)), source
            assert source in ('中', '中文') or not CHINESE.search(target), source
            entries[source] = target
    workbench = contexts['Workbench']
    for path in (ROOT / 'src/app/src').iterdir():
        if path.suffix not in ('.cpp', '.h'):
            continue
        contents = path.read_text(encoding='utf-8-sig')
        for match in LITERAL.finditer(contents):
            source = ''.join(json.loads(part) for part in re.findall(r'"(?:[^"\\]|\\.)*"', match.group(2)))
            if not CHINESE.search(source) or source in ('中', '中文'):
                continue
            assert source in workbench, (path.name, source)
            if match.group(1) == 'QStringLiteral':
                assert contents[:match.start()].endswith('aitrain_app::workbenchText('), (path.name, source)
