from rl_env.states import Message 

m1 = Message(sender="critic.1", topic="critique", content="Too short.", meta={"session_id": "S42"})
m2 = m1.fork(sender="moderator", topic="moderator_cmd", content="revise draft", receiver="author")

print(m2.sender)      # "moderator"
print(m2.receiver)    # "author"
print(m2.meta)    