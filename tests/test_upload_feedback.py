"""Exercise the browser-side state change without making a network upload."""

import shutil
import subprocess
from pathlib import Path
from unittest import TestCase


SCRIPT = Path(__file__).resolve().parents[1] / "app/templates/_upload_feedback_script.html"


class UploadFeedbackTest(TestCase):
    def test_submit_shows_progress_blocks_repeat_and_back_navigation_resets(self):
        if not shutil.which("node"):
            self.skipTest("Node.js is required for the browser script test")

        javascript = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(process.argv[1], 'utf8').replace(/<\/?script>/g, '');
const pageListeners = {};
function makeForm(hasButton) {
  const listeners = {};
  const button = hasButton ? { disabled: false } : null;
  const status = { hidden: true };
  const classes = new Set();
  const attributes = {};
  return {
    dataset: {}, button, status, attributes,
    classList: { add: (name) => classes.add(name), remove: (name) => classes.delete(name), contains: (name) => classes.has(name) },
    querySelector: (selector) => selector === '[data-upload-status]' ? status : button,
    addEventListener: (name, callback) => { listeners[name] = callback; },
    setAttribute: (name, value) => { attributes[name] = value; },
    removeAttribute: (name) => { delete attributes[name]; },
    submit() {
      const event = { defaultPrevented: false, preventDefault() { this.defaultPrevented = true; } };
      listeners.submit(event);
      return event;
    },
  };
}
const forms = [makeForm(false), makeForm(true)];
vm.runInNewContext(source, {
  document: { querySelectorAll: () => forms },
  window: { addEventListener: (name, callback) => { pageListeners[name] = callback; } },
});
for (const form of forms) {
  assert.equal(form.submit().defaultPrevented, false);
  assert.equal(form.status.hidden, false);
  assert.equal(form.dataset.uploading, 'true');
  assert.equal(form.attributes['aria-busy'], undefined);
  if (form.button) assert.equal(form.button.disabled, true);
  assert.equal(form.classList.contains('is-uploading'), true);
  assert.equal(form.submit().defaultPrevented, true);
}
pageListeners.pageshow();
for (const form of forms) {
  assert.equal(form.status.hidden, true);
  if (form.button) assert.equal(form.button.disabled, false);
  assert.equal(form.dataset.uploading, 'false');
  assert.equal(form.submit().defaultPrevented, false);
}
"""
        result = subprocess.run(
            ["node", "-e", javascript, str(SCRIPT)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
