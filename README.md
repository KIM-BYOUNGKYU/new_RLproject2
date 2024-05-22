#https://github.com/cmontalvo251/aerospace/blob/main/rockets/rocket_seminar_series/two_stage_rocket_w_aerodynamics.py

# action space
- 5개의 엔진 축에 수직방향 각속도 (-30,0,30)
- 5개의 엔진 축방향 각속도 (-30,0,30)
- 5개의 엔진 추력(min~max) => continuous action space
- 3개의 엔진 축에 수직방향 각속도 (-30,0,30)
- 3개의 엔진 축방향 각속도 (-30,0,30)
- 3개의 엔진 추력(min~max) => continuous action space
- 1개의 분리(버튼)  값이 1이상이면 1단분리, 값이 2이상이면 2단 분리
=> 시간이나 현재 연료 잔량에 따라 분리되도록 설정
=> 총 25개의 값을 list로 return


# state space
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

---

5.9 김병규 작업한 부분:
rocket.py : 
- method : init, step, get_New_state, get_aerofriction, get_gravity, get_propulsion, step 
- 부족한 부분 : 
1. 실제로 동작하는 여부는 테스트 하지 못함. 
2. get_New_state에서 coordinate 변화를 고려하지 않음. 따라서 6DOF라는 모듈사용해서 coordinate 변화 계산하기 편하게 바꿀 거임.


5.11 남윤호 작업
rocket.py
- 상수값 채우기
- 부족한 부분: TODO 참조

5.13 남민영 작업
rocket.py
- flatten 함수 제작, create_initial_state 수정
- 부족한 부분: 수정된 create_initial_state에 맞춰 get_new_state 수정 필요(y축 도입)

5.14 김병규 작업
rocket.py
- dynamics 검토 후 최고속도 O km/s 으로 나타나는 것 확인 => stage 분리작업도 추후 확인 예정
- animation 변경
test_rocket.py
- rocket 변경 후 세부사항 수정

5.18 남윤호 작업
rocket.py
- rocket.mass 수정, 관성모멘트 공식 도입
- TODO: mass에 따라 관성모멘트가 달라지는 부분도 step에 넣어야 할 듯
- 월요일 수업 전까지 policy.py 검토 예정

5.19 남민영 작업
rocket.py
- calculate_reward 함수 마무리, check_crash 함수 수정
- TODO: target_p (목표 궤도) 설정
policy.py
- ActorCritic 참고 자료: https://wikidocs.net/172977
test_train.py 생성

5.20 남윤호 작업
policy.py
- 함수 검토 완료
- 변수들의 파라미터화를 통해 테스트를 조금 더 쉽도록 함

5.20 남민영 작업
policy.py
- 함수 검토 및 주석 추가

5.22 남민영 작업
rocket.py
- target_p 변수 추가 및 calculate_reward 수정
- target_p 참고 자료: https://forum.nasaspaceflight.com/index.php?action=dlattach;topic=29214.0;attach=489078



=>>>여기다가 본인이 작업한 부분 추가해서 넣으면 될 듯합니다.


---

# What we need to: 
(구현 필요해 보이는 부분 추가하기, 완료했으면 취소선으로 변경)


 ~* flatten 함수 구현 : line45 


 7개의 요소의 list를 input으로 받아서 30개의 element를 갖는 list로 풀어주는 함수 


이 함수를 구현해야 나중에 NN에 state를 input으로 집어넣기 쉬울 것임.~

* 기본적인 변수들을 실제값으로 변경.
* 그래프 그리는 method 만들고 실제 동작 확인.
* calculate_reward 부분 구현.
* policy.py 검토 및 전체적인 RL 구성하기
* visualization: 제대로 학습 되었는지 확인하는 부분 첨가
* get_new_state: y좌표 계산 필요

