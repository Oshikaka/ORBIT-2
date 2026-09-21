# ORBIT-2 · Forecasting pipeline (published page)

This branch exists only to serve one page through GitHub Pages:

**https://oshikaka.github.io/ORBIT-2/**

`index.html` is a copy of `docs/forecasting_pipeline.html` from the
`frontier-setup` branch. It is a single self-contained file — no assets,
no build step, no external requests.

To republish after editing the source:

```sh
git switch frontier-setup            # edit docs/forecasting_pipeline.html there
git switch gh-pages
git checkout frontier-setup -- docs/forecasting_pipeline.html
mv docs/forecasting_pipeline.html index.html && rmdir -p docs 2>/dev/null
git commit -am "Republish forecasting pipeline page" && git push mine gh-pages
```

`.nojekyll` keeps GitHub Pages from running the file through Jekyll.
