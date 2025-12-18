# Deployment

We're going to implement deployment for the web app in src/app - please write a script called src/app/deploy.py that does the following things:

- builds the static web JS/HTML/CSS in src/app for serving 
- pushes that content to a github pages branch
- publishes the public pmtiles to cloudflare R2 bucket called `isometric-nyc`

Get all of the scaffolding in place first, then we'll update the scripts to have the correct API keys/permissions/etc
