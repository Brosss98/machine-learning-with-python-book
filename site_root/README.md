# site_root

Files placed here are copied, as they are, to the **root of the published site** every time you run
`python tools/book.py build` (this README is not copied). Typical use: the verification file that
Google Search Console asks you to upload (a file named like `google1a2b3c4d5e6f7890.html`).

Keep the file here for as long as you want the site to stay verified, and commit it with the repository.
Files in this folder are left out of `sitemap.xml`.
