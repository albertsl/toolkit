from urllib import robotparser

def prepare(robot_parser, robots_txt_url):
    robot_parser.set_url(robots_txt_url)
    robot_parser.read()

def is_allowed(robot_parser, target_url, user_agent='*'):
    return robot_parser.can_fetch(user_agent, target_url)

if __name__ == '__main__':
    robot_parser = robotparser.RobotFileParser()
    prepare(robot_parser, 'http://www.apress.com/robots.txt')

    print(is_allowed('http://www.apress.com/covers'))
    print(is_allowed('http://www.apress.com/gp/python'))