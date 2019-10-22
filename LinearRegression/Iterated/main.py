from numpy import *


# Error or loss function
def compute_error (
        theta_0_current,
        theta_1_current,
        points
):
    total_error = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        total_error += (y - (theta_0_current + theta_1_current * x)) ** 2
    return total_error / float(len(points))


def step_gradient (
        theta_0_current,
        theta_1_current,
        points,
        learning_rate
):
    theta_0_gradient = 0
    theta_1_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        theta_0_gradient += -(2 / N) * (y - ((theta_1_current * x) + theta_0_current))
        theta_1_gradient += -(2 / N) * x * (y - ((theta_1_current * x) + theta_0_current))

    new_theta_0 = theta_0_current - (learning_rate * theta_0_gradient)
    new_theta_1 = theta_1_current - (learning_rate * theta_1_gradient)
    return new_theta_0, new_theta_1


def gradient_descent_runner (
        points,
        starting_theta_0,
        starting_theta_1,
        learning_rate,
        num_iterations
):
    theta_0 = starting_theta_0
    theta_1 = starting_theta_1

    for i in range(num_iterations):
        theta_0, theta_1 = step_gradient(
            theta_0,
            theta_1,
            array(points),
            learning_rate
        )
    return theta_0, theta_1


def main ():
    points = genfromtxt('data.csv', delimiter=',')
    # hyperparameters
    learning_rate = 0.0001
    num_iterations = 1000
    # y = theta_0 + theta_1 * x (slope formula )
    initial_theta_0 = 0
    initial_theta_1 = 0
    print ("Starting gradient descent:\ntheta_0 = {0}, theta_1 = {1}, error = {2}".format (
        initial_theta_0, initial_theta_1, compute_error(initial_theta_0, initial_theta_1, points)))
    [final_theta_0, final_theta_1] = gradient_descent_runner(
        points,
        initial_theta_0,
        initial_theta_1,
        learning_rate,
        num_iterations
    )
    print("After gradient descent: \ntheta_0 = {0}, theta_1 = {1}, error = {2}".format (
        final_theta_0,
        final_theta_1,
        compute_error(final_theta_0, final_theta_1, points)
    ))


if __name__ == "__main__":
    main()
