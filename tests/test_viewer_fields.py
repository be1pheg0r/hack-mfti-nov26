"""Regression checks for record-specific fields in the real viewer script."""

import subprocess
from pathlib import Path


def test_overview_shows_actual_fields_without_inventing_vacancy_fields():
    """A resume has no invented requirements; real vacancy fields remain visible."""
    root = Path(__file__).resolve().parents[1]
    script = r"""
const fs = require('fs');
const vm = require('vm');
const assert = require('assert');
class Element {
  constructor(tag) { this.tag = tag; this.children = []; this.textContent = ''; }
  append(...nodes) { this.children.push(...nodes); }
  setAttribute() {}
}
const context = { document: { createElement: tag => new Element(tag) } };
vm.createContext(context);
const source = fs.readFileSync('web/dataset_viewer/app.js', 'utf8');
assert(source.includes('  init();\n})();'));
vm.runInContext(source.replace('  init();\n})();',
  '  globalThis.renderForTest = renderOverview;\n})();'), context);
function fields(record) {
  const panel = new Element('div');
  context.renderForTest(record, panel);
  const grid = panel.children.find(node => node.className === 'field-grid');
  return Object.fromEntries(grid.children.map(node =>
    [node.children[0].textContent, node.children[1].textContent]));
}
const resume = fields({kind: 'resume', text: '', attributes: {
  experience: 7, education: 'Высшее-бакалавриат', salaryMin: 270000
}});
assert.strictEqual(resume['Опыт работы'], '7');
assert.strictEqual(resume['Образование'], 'Высшее-бакалавриат');
assert(!('Требуемый опыт' in resume));
assert(!('Требования к образованию' in resume));
assert(!('Занятость' in resume));
assert(!('Обязанности' in resume));
assert(!('Требования к должности' in resume));
const vacancy = fields({kind: 'vacancy', text: '', attributes: {
  experienceRequirements: 3, educationRequirements: 'Высшее',
  busyType: 'Полная занятость', responsibilities: 'Разработка',
  positionRequirements: 'Python'
}});
assert.strictEqual(vacancy['Требуемый опыт'], '3');
assert.strictEqual(vacancy['Требования к образованию'], 'Высшее');
assert.strictEqual(vacancy['Занятость'], 'Полная занятость');
assert.strictEqual(vacancy['Обязанности'], 'Разработка');
assert.strictEqual(vacancy['Требования к должности'], 'Python');
"""
    subprocess.run(["node", "-e", script], cwd=root, check=True, timeout=5)
