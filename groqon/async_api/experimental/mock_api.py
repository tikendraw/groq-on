from playwright.async_api import Page, async_playwright, Route
from ..agroq import groq_context
from ...groq_config import API_URL, URL

PAYLOAD = {
  "model": "llama3-8b-8192",
  "messages": [
    {
      "content": "Please try to provide useful, helpful and actionable answers.",
      "role": "system"
    },
    {
      "content": "how old is sun",
      "role": "user"
    }
  ],
  "temperature": 0.2,
  "max_tokens": 2048,
  "top_p": 1,
  "stream": True
}

async def test_mock_the_fruit_api(page: Page):
    def handle(route: Route):
        json = [PAYLOAD]
        # fulfill the route with the mock data
        route.fulfill(json=json)

    # Intercept the route to the fruit API
    await page.route(API_URL, handle)
    await page.wait_for_timeout(100*1000)
    await page.screenshot(path="after_routing.png")
    # Go to the page
    await page.goto(URL)
    await page.screenshot(path="after_going.png")

async def main():
    async with groq_context() as context:
        page = await context.new_page()
        await page.goto(URL)
        await page.screenshot(path="before_routing.png")
        await test_mock_the_fruit_api(page)
        await page.screenshot(path="after_testing.png")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
