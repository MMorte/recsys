from ncf.data import read_books, train_test_split
from ncf.model import CollaborativeFiltering


if __name__ == "__main__":
    data = read_books()
    train, test = train_test_split(data=data)
    cf = CollaborativeFiltering(
        learning_rate=1e-2,
        n_factors=32,
        n_epochs=1,
        batch_size=128,
        y_range=(0.0, 10.0),
    )
    cf.fit(train, verbose=True)
    print(cf.evaluate(test))
