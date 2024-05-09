#https://github.com/cmontalvo251/aerospace/blob/main/rockets/rocket_seminar_series/two_stage_rocket_w_aerodynamics.py

*action space
- 5개의 엔진 축에 수직방향 각속도 (-30,0,30)
- 5개의 엔진 축방향 각속도 (-30,0,30)
- 5개의 엔진 추력(min~max) => continuous action space
- 3개의 엔진 축에 수직방향 각속도 (-30,0,30)
- 3개의 엔진 축방향 각속도 (-30,0,30)
- 3개의 엔진 추력(min~max) => continuous action space
- 1개의 분리(버튼)  값이 1이상이면 1단분리, 값이 2이상이면 2단 분리
=> 총 25개의 값을 list로 return


*state space
- (x, y, z 좌표)    np.array:3개 element
- (x, y, z 속도)    np.array:3개 element
- (회전 각도)        np.array:3개 element
- (회전 각속도)     np.array:3개 element
- 현재 연료 질량
- 현재 stage
- 선체를 기준으로 노즐 각도 ((0theta0, 0theta1), (1theta0, 1theta1), (2theta0, 2theta1), (3theta0, 3theta1), ...) 2D array: 2*8개 element
    theta0 = 0\~max
    theta1 = 0\~2pi
=> 30개의 element를 state에 저장
=> 7개의 요소를 list로 저장



5.9 김병규 작업한 부분:
rocket.py : 
- method : init, step, get_New_state, get_aerofriction, get_gravity, get_propulsion, step 
- 부족한 부분 : 
1. 실제로 동작하는 여부는 테스트 하지 못함. 
2. get_New_state에서 coordinate 변화를 고려하지 않음. 따라서 6DOF라는 모듈사용해서 coordinate 변화 계산하기 편하게 바꿀 거임.

=>>>여기다가 본인이 작업한 부분 추가해서 넣으면 될 듯합니다.

What we need to: (구현 필요해 보이는 부분 추가하기, 완료했으면 취소선으로 변경)
*flatten 함수 구현 : line45
7개의 요소의 list를 input으로 받아서 30개의 element를 갖는 list로 풀어주는 함수
이 함수를 구현해야 나중에 NN에 state를 input으로 집어넣기 쉬울 것임. 
*기본적인 변수들을 실제값으로 변경.
*calculate_reward 부분 구현.
*policy.py 검토 및 전체적인 RL 구성하기
*visualization: 제대로 학습 되었는지 확인하는 부분 첨가

