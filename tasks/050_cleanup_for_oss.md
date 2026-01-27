# Clean Up for OSS

I want to clean up this repo for releasing it to open source. Here are a few things I want to do:

## Make all scripts/tasks consistent
Let's make all python scripts have entries in the pyproject.toml and ensure that they're documented. Ensure that various web apps and web tasks are also cleaned up, documented, and consistent.

## Clean up old scripts/workflows
I need to clean up scripts and workflows that are no longer relevant - these include things with the original 3D city data, although we can keep the "whitebox" flow and accompanying scripts around for historical context.

## Documentation
Let's ensure that README.md is up-to-date. This means how to start the entire project from github to working generation app, including what env variables to set, how to get the oxen data, how to set up inference with a model on oxen *and* on lambda labs (code now added to /inference). I also want to run through the entire workflow of generating tiles and/or fixing them, and how to run the local web app and how it's deployed to cloudflare (but no actual deployment possible via the OSS repo)

## Scrubbing secrets / flattening

I want to remove all possible secrets from the repo - this includes my personal info (directories) and especially any hard-coded API keys or tokens. I also want to eventually flatten all of the commit history into one commit before deploying and releasing.

## Plan

<INSERT PLAN HERE>
