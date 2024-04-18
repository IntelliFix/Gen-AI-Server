from unstructured.partition.html import partition_html

cnn_lite_url = "https://python.langchain.com/docs/get_started/introduction"
elements = partition_html(url=cnn_lite_url)
links = []

for element in elements:
    if element.metadata.link_urls:
        relative_link = element.metadata.link_urls[0][1:]
        if relative_link.startswith("docs"):
            links.append(f"https://python.langchain.com/{relative_link}")
            url = "https://python.langchain.com/" + relative_link
            partitions = partition_html(url=url)
            for partition in partitions:
                if partition.metadata.link_urls:
                    relative_link = partition.metadata.link_urls[0][1:]
                    if relative_link.startswith("docs"):
                        links.append(f"https://python.langchain.com/{relative_link}")

print(len(links))