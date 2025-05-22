import logo from 'images/smarter_rm_bg.png'

export const Header = () => (
  <div className="flex flex-row justify-between space-x-6 px-8 py-3.5 bg-black w-full">
    <div className="pl-8 border-l border-ink">
      <a href="/">
        <img width={118} height={40} src={logo} alt="Logo" />
      </a>
    </div>
  </div>
)
