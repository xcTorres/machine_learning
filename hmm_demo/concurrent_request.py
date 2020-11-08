# @Time    : 15/10/20 15:09
# @Author  :  xcTorres
# @FileName: concurrent_request.py


import asyncio
from aiohttp import ClientSession, TCPConnector


async def create_session():
    """Create session
    """
    conn = TCPConnector(limit=30)
    session = ClientSession(connector=conn)
    return session


async def async_request(session, request_url, params):
    """Async route engine request"""
    try:
        async with session.post(request_url, json=params) as response:
            return await response.json(content_type=None)
    except:
        return None


async def gather_tasks(tasks):
    """Gather tasks"""
    return await asyncio.gather(*tasks)


def fetch(base_url, params):
    task_list = []
    loop = asyncio.new_event_loop()
    session = loop.run_until_complete(create_session())
    for p in params:
        task = async_request(session, base_url, p)
        task_list.append(task)

    response = loop.run_until_complete(gather_tasks(task_list))
    loop.run_until_complete(session.close())
    loop.close()

    return response