# Third-party notices

## Flask documentation (the demo corpus)

This repository and the results site (`site/data.json`, `eval/*.json`, `eval/anythingllm_example.*`)
reproduce excerpts of the Flask documentation, including its code examples (for example the
placeholder `SECRET_KEY` values shown in Flask's own docs; they are upstream examples, not
credentials of this project). Those excerpts are not covered by this project's MIT license.

- Source: https://flask.palletsprojects.com/en/stable/ (rendered pages titled "Flask Documentation (3.1.x)")
- Upstream repository: https://github.com/pallets/flask (latest 3.1.x release at extraction time: 3.1.3)
- Extracted: 2026-10-05, by this repo's crawler (74 pages; see `data/raw/manifest.json` when you run it)
- License text below copied verbatim from https://github.com/pallets/flask/blob/3.1.3/LICENSE.txt
  (identical to the file on `main` on 2026-10-05)

This project is not affiliated with or endorsed by Pallets.

```text
Copyright 2010 Pallets

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

1.  Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

2.  Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.

3.  Neither the name of the copyright holder nor the names of its
    contributors may be used to endorse or promote products derived from
    this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
