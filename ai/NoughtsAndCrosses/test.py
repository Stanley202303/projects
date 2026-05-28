
# print(str([1, 2, 3]))
# import json, functools, time
# with open("ai/NoughtsAndCrosses/data.json", 'r') as data:
#     with open("ai/NoughtsAndCrosses/data2.json", 'r') as data2:
#         print(json.load(data) == json.load(data2))

# with open("ai/NoughtsAndCrosses/data.json", 'r') as data:
#     with open("ai/NoughtsAndCrosses/data2.json", 'w') as data2:
#         # with open('ai/NoughtsAndCrosses/backup.json', 'x') as backup:
#             json.dump(json.load(data), data2, indent=4)

# @functools.lru_cache(256)
# def fibonacci(n):
#     if n < 3:
#         return 1
#     return fibonacci(n-1) + fibonacci(n-2)

# def bad_fib(n):
#     if n < 3:
#         return 1
#     else:
#         return bad_fib(n-1) + bad_fib(n-2)
    
# def measure_time(function, *args):
#     start = time.time()
#     function(*args)
#     end = time.time()
#     print(end-start)

# measure_time(fibonacci, 40)
# print('running bad_fib')
# measure_time(bad_fib, 40)



