import hashlib
import json
import random

adminlogin = False

try:
    with open('admindetails.txt', 'r') as f:
        details = f.readline()
except:
    print("Welcome, Please create admin account")
    username = input("enter username: ")
    password = input("enter password: ")
    repassword = input("Re-Enter Password: ")
    while repassword != password:
        print("Password Doesnt Match")
        password = input("Password: ")
        repassword = input("Re-Enter Password: ")
    res = hashlib.sha256(password.encode())
    password = res.hexdigest()

    with open('admindetails.txt', 'w') as f:
        f.write(username + ":" + password)
    print('Admin account created successfully')


def login(username, password):
    global adminlogin
    with open('admindetails.txt', 'r') as f:
        details = f.readline()
        adminusername = details.split(':')[0]
        adminpass = details.split(':')[1]
    if adminusername == username and adminpass == password:
        adminlogin = True
        return True
    with open('userdetails.txt', 'a+') as f:
        f.seek(0)
        lines = f.readlines()
        for details in lines:
            if details.split(':')[0] == username and details.split(':')[1] == password:
                adminlogin = False
                return True
    return False

class User:
    def __init__(self,username):
        self.username = username
        self.QuizPaper = {}
        self.scorelist = {}
        self.count = 0
        self.score = 0
        try:
            with open('QuizPaper.txt', 'r') as f:
                self.QuizPaper = json.load(f)

            with open('QuizQuestionCount.txt', 'r') as f:
                self.count = int(f.readline())
        except:
            pass
        while True:
            option = input("Enter \n 1) Write Quiz  \n 2) My Score. \n enter 0 to exit the application: ")
            if option == '0':
                return
            if option == '1':
                self.WriteQuiz()
            if option == '2':
                print('you last quiz score is:',self.scorelist[self.username])
    def WriteQuiz(self):
        temp = self.QuizPaper.copy()
        if self.count > 0:
            qno = 1
            while qno <= self.count:
                for ques in self.QuizPaper:
                    if qno > self.count:
                        break
                    if random.randint(0,1) == 1:
                        print("{0}) {1}".format(qno,ques))
                        qno += 1
                        ans = ""
                        if len(self.QuizPaper[ques]) == 1:
                            print("1) True")
                            print("2) False")
                            choice = input("enter option no (1,2): ")
                            if choice == '1':
                               ans = 'true'
                            if choice == '2':
                                ans = 'false'
                            if self.QuizPaper[ques][0].lower() == ans:
                                self.score += 1
                        else:
                            print("1) {0}".format(self.QuizPaper[ques][0]))
                            print("2) {0}".format(self.QuizPaper[ques][1]))
                            print("3) {0}".format(self.QuizPaper[ques][2]))
                            print("4) {0}".format(self.QuizPaper[ques][3]))
                            choice = input("enter option no (1,2.3.4): ")

                            if self.QuizPaper[ques][4].lower() == choice:
                                self.score += 1
            print("test completed")
            self.scorelist[self.username] = self.score
            with open('scorelist.txt','w') as f:
                json.dump(self.scorelist,f)

        else:
            print("quiz questions are not updated, please contact admin")
        self.QuizPaper = temp
    @staticmethod
    def CreateUser():
        loginusername = input("username: ")
        mobile = input("Phone Number: ")
        email = input("Email: ")
        loginpassword = input("password: ")
        repassword = input("Re-Enter password: ")
        while repassword != loginpassword:
            print("loginpassword Doesnt Match")
            loginpassword = input("password: ")
            repassword = input("Re-Enter password: ")
        res = hashlib.sha256(loginpassword.encode())
        password = res.hexdigest()
        with open('userdetails.txt','a') as f:
            f.write("{0}:{1}:{2}:{3}\n".format(loginusername,password,email,mobile))
        print("user account created successfully")
class Admin:
    def __init__(self):
        self.QuizPaper = {}
        try:
            with open('QuizPaper.txt', 'r') as f:
                self.QuizPaper = json.load(f)
        except:
            pass
        while True:
            option = input("Enter \n 1) Add Question \n 2) Remove Question. \n 3) Update Quiz Question count. \n 4) show questions \nenter 0 to exit the application: ")
            if option not in ['0', '1', '2', '3', '4']:
                print("please enter correct option")

            else:
                if option == '0':
                    return
                if option == '4':
                    self.showallquestions()
                if option == '3':
                    count = input('enter quiz question count: ')
                    with open('QuizQuestionCount.txt', 'w') as f:
                        f.write(count)
                    print("QuizQuestionCount Updated Successfully")
                if option == '1':
                    ans = []
                    ques = input("enter Question:- ")
                    type = input("1) True/False Type \n2) MCQ Type\n")

                    if type == '2':
                        ans.append(input("enter Option 1: "))
                        ans.append(input("enter Option 2: "))
                        ans.append(input("enter Option 3: "))
                        ans.append(input("enter Option 4: "))
                        ans.append(input("Correct Option (enter 1,2,3,4):- "))
                        self.QuizPaper[ques] = ans
                        self.UpdateFileDic()

                    if type == '1':
                        ans.append(input("Correct Option enter True/False: "))
                        self.QuizPaper[ques] = ans
                        self.UpdateFileDic()

                if option == '2':
                    if len(self.QuizPaper) < 1:
                        print("No Questions")
                    else:
                        temp = []
                        for ques in self.QuizPaper:
                            temp.append(ques)
                        for i in range(len(temp)):
                            print("{0}) {1}".format(i + 1, temp[i]))
                        quesid = int(input("Choose question No which you wanted to remove,press 0 to delete all:- "))
                        if quesid > 0:
                            if input("enter 1 to remove question :- %s" % temp[quesid - 1]) == '1':
                                del self.QuizPaper[temp[quesid - 1]]
                            self.UpdateFileDic()
                        if quesid == 0:
                            self.QuizPaper = {}
                            self.UpdateFileDic()
            input('enter to continue....')
    def UpdateFileDic(self):
        with open('QuizPaper.txt', 'w') as f:
            json.dump(self.QuizPaper, f)
            print("QuizPaper Updated Successfully")

    def showallquestions(self):
        qno = 1
        for ques in self.QuizPaper:
            print("{0}) {1}".format(qno,ques))
            qno += 1

while True:
    try:
        switch = int(
            input("Enter \n 1) admin or user login. \n 2) Create new user. \n enter 0 to exit the application: "))
        if switch not in [0, 1, 2]:
            raise "wrong input entered"
        if switch == 0:
            break
        elif switch == 2:
            User.CreateUser()
        elif switch == 1:
            username = input("Enter UserName: ")
            res = hashlib.sha256(input("Enter Password: ").encode())
            password = res.hexdigest()
            if login(username, password):
                while True:
                    print("Login Successful")
                    try:
                        if adminlogin:
                            admin = Admin()
                        else:
                            user = User(username)
                        break
                    except Exception as ex:
                        print(ex.args)
                        break
            else:
                print("Log in Failed, Invalid Credentials")
                input('press enter to continue...')
    except:
        print("option invalid, enter only 1 or 2")
        input('press enter to continue...')
