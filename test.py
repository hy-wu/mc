import asyncio


async def simu(i):
    print(f"simu({i}) 开始模拟")
    await asyncio.sleep(10)  # 模拟实际模拟操作的耗时
    print(f"第{i}个任务已结束")


async def main():
    tasks = []
    for i in range(1, 11):
        # await asyncio.sleep(1)  # 每隔1秒启动一个新的模拟
        task = asyncio.create_task(simu(i))
        tasks.append(task)
        print(f'Start running config {i}')
        await asyncio.sleep(1)
    await asyncio.gather(*tasks)
    print("所有模拟已完成")


if __name__ == "__main__":
    asyncio.run(main())