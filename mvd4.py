from datetime import datetime

import pandas as pd
import vk_api

app_id = 0
token = "0"

session : vk_api.VkApi = vk_api.VkApi(
    app_id=app_id,
    token=token
)
api = session.get_api()


post_number = 500
iters = (post_number // 100)
if iters != 500 / 100:
    iters += 1

dfs = []

for i in range(iters):
    batch = api.wall.get(domain="kinoz", count=100, offset=i*100)
    posts = batch["items"]

    texts = []
    dates = []

    for post in posts:
        text = post["text"]
        date = post["date"]
        formatted_date = str(datetime.fromtimestamp(date))

        texts.append(text)
        dates.append(formatted_date)

    df = pd.DataFrame(data={"text": texts, "date": dates})
    dfs.append(df)

main_df = pd.concat(dfs, ignore_index=True)


