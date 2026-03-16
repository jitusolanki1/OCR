# RENDER DEPLOYMENT MANUAL CONFIGURATION REQUIRED

## To fix the Python 3.14 buildpack issue:

1. Go to https://dashboard.render.com
2. Click your **ocr-api** service
3. Go to **Settings** tab
4. Scroll down to **Build Command** section
5. Change **Build** from **Native** to **Docker**
6. Save the changes
7. Click **Deploy** to trigger a new build

Render will now:

- Use the Dockerfile (Python 3.12)
- Install all dependencies with pre-built wheels
- Successfully deploy without compilation errors

If you don't see a "Docker" option:

- Delete the service and create a new one linking to your GitHub repo
- When prompted for "Build", select **Docker**
