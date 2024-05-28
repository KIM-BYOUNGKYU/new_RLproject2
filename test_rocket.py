import rocket

falcon9 = rocket.Rocket(max_steps=400000)
action = []
for i in range(5):
    action.append(1)
for i in range(5):
    action.append(1)
for i in range(5):
    action.append(falcon9.max_thrust[0])
for i in range(3):
    action.append(0)
for i in range(3):
    action.append(0)
for i in range(3):
    action.append(falcon9.max_thrust[0])


falcon9.step(action)
for i in range(100000):
    state,reward,done,_=falcon9.step(action)
    if done:
        break

falcon9.show_path_from_state_buffer()
falcon9.animate_trajectory(skip_steps=100)